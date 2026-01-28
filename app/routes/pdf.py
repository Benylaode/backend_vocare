import pymupdf
import easyocr
from PIL import Image
import io
import numpy as np
import os
from flask import Blueprint, request, jsonify
import faiss
from sentence_transformers import SentenceTransformer
from app.utils import role_required
from flask_jwt_extended import jwt_required
import pickle
import re

# Init model
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    reader = easyocr.Reader(['id', 'en'])
except ImportError as e:
    print(f"Error initializing models: {e}. Please ensure you have sentence-transformers and easyocr installed.")
    model = None
    reader = None

document_bp = Blueprint('document_bp', __name__, url_prefix='/pdf')

FAISS_DIR = os.path.join(os.path.dirname(__file__), ".", "faisses")
os.makedirs(FAISS_DIR, exist_ok=True)

EMBEDDING_DIM = model.get_sentence_embedding_dimension() if model else 384


# === Utility Functions ===

def initialize_faiss_index(file_path: str):
    """Load FAISS index kalau ada, kalau tidak buat baru."""
    if os.path.exists(file_path):
        return faiss.read_index(file_path)
    return faiss.IndexIDMap(faiss.IndexFlatL2(EMBEDDING_DIM))

def save_faiss_index(index, file_path: str):
    """Simpan FAISS index ke disk."""
    faiss.write_index(index, file_path)

def save_mapping(mapping, file_path: str):
    """Simpan mapping id→teks ke pickle."""
    with open(file_path, "wb") as f:
        pickle.dump(mapping, f)

def load_mapping(file_path: str):
    """Load mapping id→teks kalau ada, kalau tidak return dict kosong."""
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return {}

def extract_text_from_pdf(pdf_path: str):
    """Ekstrak teks dengan menjaga struktur layout."""
    text_content = []
    try:
        with pymupdf.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                # Gunakan "blocks" untuk memisahkan header/footer dan paragraf
                # Blocks return: (x0, y0, x1, y1, "text", block_no, block_type)
                blocks = page.get_text("blocks")
                page_text = ""
                
                for b in blocks:
                    if b[6] == 0:  # Tipe 0 adalah teks
                        cleaned_block = b[4].strip()
                        if cleaned_block:
                            page_text += cleaned_block + "\n\n" # Pisah antar blok dengan 2 enter
                
                if page_text.strip():
                    text_content.append(f"--- Halaman {page_num} ---\n{page_text}")
                
                # Fallback ke OCR jika halaman kosong (Gambar)
                elif reader: 
                    pix = page.get_pixmap()
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    img_np = np.array(img)
                    ocr_result = reader.readtext(img_np, detail=0)
                    ocr_text = "\n".join(ocr_result)
                    if ocr_text.strip():
                        text_content.append(f"--- Halaman {page_num} (OCR) ---\n{ocr_text}")
                        
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None
        
    return "\n".join(text_content)

import re  # Pastikan ini ada di paling atas file

def split_by_no(text):
    """
    Memecah dokumen berdasarkan nomor diagnosis (NO 1, NO 2, dst)
    Return: list of dict {no, content}
    """
    pattern = r"\n\s*(\d+)\s+(?=[A-Z])"
    parts = re.split(pattern, text)

    rows = []
    for i in range(1, len(parts), 2):
        no = parts[i]
        content = parts[i + 1].strip()
        rows.append({
            "no": no,
            "text": f"NO {no}\n{content}"
        })
    return rows


def build_indexes_from_rows(rows):
    symptom_rows = []
    full_rows = []

    for r in rows:
        text = r["text"]

        # PERBAIKAN: Gunakan re.IGNORECASE dan pola yang lebih fleksibel
        # Menangkap apa saja setelah kata kunci sampai ketemu kata kunci berikutnya atau akhir baris
        
        # 1. Ambil Data Subjektif
        subj_match = re.search(
            r"(?:DATA SUBJEKTIF|Data Subjektif)\s*[:\n]\s*([\s\S]*?)(?=(?:DATA OBJEKTIF|Data Objektif)|$)", 
            text, 
            re.IGNORECASE
        )
        
        # 2. Ambil Data Objektif
        obj_match = re.search(
            r"(?:DATA OBJEKTIF|Data Objektif)\s*[:\n]\s*([\s\S]*?)(?=\n\d+\s+[A-Z]|$)", 
            text, 
            re.IGNORECASE
        )

        subj_text = subj_match.group(1).strip() if subj_match else "-"
        obj_text = obj_match.group(1).strip() if obj_match else "-"

        # Buat teks khusus untuk Index Pencarian (Symptom Index)
        # Kita gabungkan agar FAISS fokus mencocokkan keluhan user ke sini
        symptom_text = f"KELUHAN PASIEN (Subjektif): {subj_text}\nTEMUAN KLINIS (Objektif): {obj_text}"

        symptom_rows.append({
            "no": r["no"],
            "text": symptom_text  # Ini yang akan di-embedding untuk pencarian
        })

        # Full row tetap utuh untuk context generation nanti
        full_rows.append(r)

    return symptom_rows, full_rows
def build_faiss_from_rows(rows, faiss_path, pkl_path):
    index = faiss.IndexIDMap(faiss.IndexFlatL2(EMBEDDING_DIM))
    mapping = {}

    for i, row in enumerate(rows):
        emb = model.encode([row["text"]]).astype("float32")
        index.add_with_ids(emb, np.array([i]))

        mapping[i] = {
            "no": row["no"],
            "text": row["text"]
        }

    faiss.write_index(index, faiss_path)
    with open(pkl_path, "wb") as f:
        pickle.dump(mapping, f)



def row_chunk_text(text):
    """
    Memecah dokumen berdasarkan BARIS TABEL (NO 1, NO 2, dst)
    sehingga 1 embedding = 1 diagnosis SDKI lengkap
    """
    pattern = r"\n\s*(\d+)\s+(?=[A-Z])"
    
    splits = re.split(pattern, text)
    
    chunks = []
    for i in range(1, len(splits), 2):
        no = splits[i]
        content = splits[i + 1]
        chunk = f"NO {no}\n{content.strip()}"
        chunks.append(chunk)

    return chunks


def process_pdf_and_save(file, file_type: str, faiss_dir: str):
    """
    Dual Index Builder:
    - Index A: symptom.faiss → Data Subjektif + Objektif (untuk retrieval relevansi)
    - Index B: full.faiss → Full SDKI + SIKI + SLKI per NO (untuk jawaban utuh)
    """
    if not model or not reader:
        return {'error': 'Model tidak diinisialisasi'}, 500

    os.makedirs(faiss_dir, exist_ok=True)
    temp_filepath = os.path.join(faiss_dir, file.filename)

    try:
        file.save(temp_filepath)

        extracted_text = extract_text_from_pdf(temp_filepath)
        if not extracted_text:
            return {'error': 'Gagal mengekstrak teks dari PDF'}, 500

        rows = split_by_no(extracted_text)
        if not rows:
            return {'message': 'Tidak ditemukan baris diagnosa (NO) dalam PDF.'}, 200
        symptom_rows, full_rows = build_indexes_from_rows(rows)

        # Hapus index lama kalau ada
        for f in ["symptom.faiss", "symptom.pkl", "full.faiss", "full.pkl"]:
            path = os.path.join(faiss_dir, f)
            if os.path.exists(path):
                os.remove(path)

        # =====================
        # 5. SIMPAN FAISS
        # =====================
        build_faiss_from_rows(
            symptom_rows,
            os.path.join(faiss_dir, "symptom.faiss"),
            os.path.join(faiss_dir, "symptom.pkl")
        )

        build_faiss_from_rows(
            full_rows,
            os.path.join(faiss_dir, "full.faiss"),
            os.path.join(faiss_dir, "full.pkl")
        )

        return {
            "message": "PDF berhasil diproses dengan Dual Index (Gejala + Full SDKI-SIKI-SLKI)",
            "total_rows": len(rows),
            "symptom_index": "symptom.faiss",
            "full_index": "full.faiss"
        }, 200

    except Exception as e:
        return {'error': str(e)}, 500

    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)



# === ROUTES ===

@document_bp.route('/process-assesmen', methods=['POST'])
@jwt_required()
@role_required("admin", "ketim")
def process_assesmen():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file dalam permintaan'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
    resp, code = process_pdf_and_save(file, "assesmen", "app/faisses/assesment")
    return jsonify(resp), code


@document_bp.route('/process-lab', methods=['POST'])
@jwt_required()
@role_required("admin", "user")
def process_lab():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file dalam permintaan'}), 400
    file = request.files['file']
    cppt_id = request.form.get('cppt_id')
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
    if not cppt_id:
        return jsonify({'error': 'cppt_id wajib diisi'}), 400
    faiss_dir = f"app/faisses/lab_results/{cppt_id}"
    resp, code = process_pdf_and_save(file, "hasil_lab", faiss_dir)
    return jsonify(resp), code


@document_bp.route('/process-permenkes', methods=['POST'])
@jwt_required()
@role_required("admin", "ketim")
def process_permenkes():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file dalam permintaan'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
    resp, code = process_pdf_and_save(file, "permenkes", "app/faisses/permenkes")
    return jsonify(resp), code


@document_bp.route('/process-siki-slki-sdki', methods=['POST'])
@jwt_required()
@role_required("admin", "ketim")
def process_siki_slki_sdki():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file dalam permintaan'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
    resp, code = process_pdf_and_save(file, "siki-slki-sdki", "app/faisses/siki-slki-sdki")
    return jsonify(resp), code
