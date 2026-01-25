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

def chunk_text(text, chunk_size=1000, overlap=200):
    if not text:
        return []

    # DEFINISI SEPARATOR DENGAN REGEX
    # Kita pakai regex pattern untuk kalimat agar mengenali titik (.), tanya (?), seru (!)
    # Pattern r'(?<=[.?!])\s+' artinya: Potong di spasi (\s+) HANYA jika didahului . ? atau !
    separators = ["\n\n", "\n", r'(?<=[.?!])\s+', " ", ""]
    
    def split_recursive(current_text, current_separators):
        if len(current_text) <= chunk_size:
            return [current_text]
        
        if not current_separators:
            # Fallback: potong paksa karakter
            return [current_text[i:i + chunk_size] for i in range(0, len(current_text), chunk_size - overlap)]

        separator = current_separators[0]
        next_separators = current_separators[1:]
        
        # --- BAGIAN REGEX DIPAKAI DI SINI ---
        if separator == "":
            splits = list(current_text)
        elif separator == r'(?<=[.?!])\s+': 
            # Gunakan re.split untuk separator kalimat regex
            splits = re.split(separator, current_text)
        else:
            # Gunakan string split biasa untuk \n atau spasi
            splits = current_text.split(separator)
        # ------------------------------------
            
        merged_splits = []
        current_chunk = ""
        
        for split in splits:
            # Perbaikan kecil: Jika pakai regex split, separatornya sering hilang/terpisah.
            # Kita perlu menyambung ulang dengan spasi jika separatornya adalah pola kalimat.
            spacer = ""
            if separator == "\n\n": spacer = "\n\n"
            elif separator == "\n": spacer = "\n"
            elif separator == " ": spacer = " "
            # Untuk regex sentence, spasi biasanya ikut terpotong, jadi tidak perlu spacer tambahan
            # karena logic merge akan menyatukan.
            
            segment = split + spacer
            
            if len(current_chunk) + len(segment) <= chunk_size:
                current_chunk += segment
            else:
                if current_chunk:
                    merged_splits.append(current_chunk)
                current_chunk = segment
        
        if current_chunk:
            merged_splits.append(current_chunk)
            
        final_processed = []
        for chunk in merged_splits:
            if len(chunk) > chunk_size:
                final_processed.extend(split_recursive(chunk, next_separators))
            else:
                final_processed.append(chunk)
                
        return final_processed

    # Bersihkan hasil
    final_chunks = split_recursive(text, separators)
    return [c.strip() for c in final_chunks if c.strip()]

def process_pdf_and_save(file, file_type: str, faiss_dir: str):
    """Fungsi utama untuk memproses PDF dan simpan embeddings ke FAISS + mapping pkl."""
    if not model or not reader:
        return {'error': 'Model tidak diinisialisasi'}, 500

    os.makedirs(faiss_dir, exist_ok=True)
    temp_filepath = os.path.join(faiss_dir, file.filename)
    faiss_file_path = os.path.join(faiss_dir, f"{file_type}.faiss")
    pkl_file_path = os.path.join(faiss_dir, f"{file_type}.pkl")

    try:
        # Simpan file sementara
        file.save(temp_filepath)

        # Ekstraksi teks
        extracted_text = extract_text_from_pdf(temp_filepath)
        if not extracted_text:
            return {'error': 'Gagal mengekstrak teks dari PDF'}, 500

        # Chunking
        chunks = chunk_text(extracted_text)
        if not chunks:
            return {'message': 'Tidak ada teks yang dapat diproses dalam PDF.'}, 200

        # Kalau sudah ada file lama → hapus
        if os.path.exists(faiss_file_path):
            os.remove(faiss_file_path)
        if os.path.exists(pkl_file_path):
            os.remove(pkl_file_path)

        # Buat index baru
        index = faiss.IndexIDMap(faiss.IndexFlatL2(EMBEDDING_DIM))
        mapping = {}

        # Proses embeddings per batch
        batch_size = 100
        total_chunks = len(chunks)
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            embeddings = model.encode(batch_chunks)
            ids = np.arange(i, i + len(batch_chunks))
            index.add_with_ids(np.array(embeddings).astype('float32'), ids)
            for j, chunk in zip(ids, batch_chunks):
                mapping[int(j)] = chunk

        # Simpan FAISS index + mapping
        save_faiss_index(index, faiss_file_path)
        save_mapping(mapping, pkl_file_path)

        return {
            'message': f'PDF diproses ulang dan {len(chunks)} embeddings disimpan sebagai {file_type}.',
            'faiss_file': faiss_file_path,
            'pkl_file': pkl_file_path,
            'total_chunks': len(chunks)
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
