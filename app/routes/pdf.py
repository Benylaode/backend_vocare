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

FAISS_ASSESMENT_INDEX_FILE = 'assesment.faiss'
FAISS_LAB_INDEX_FILE = 'lab_results.faiss'
FAISS_PEREMENKES_INDEX_FILE = 'permenkes.faiss'
FAISS_SIKI_SLKI_SDKI_INDEX_FILE = 'siki_slki_sdki.faiss'

EMBEDDING_DIM = model.get_sentence_embedding_dimension()

def initialize_faiss_index(file_path):
    """
    Inisialisasi FAISS index atau muat dari file yang ada.
    """
    if os.path.exists(file_path):
        return faiss.read_index(file_path)
    else:
        return faiss.IndexIDMap(faiss.IndexFlatL2(EMBEDDING_DIM))

def save_faiss_index(index, file_path):
    """
    Menyimpan FAISS index ke disk.
    """
    faiss.write_index(index, file_path)

def extract_text_from_pdf(pdf_path):
    """
    Ekstrak teks dari dokumen PDF, menggunakan OCR untuk halaman berbasis gambar.
    """
    text = ""
    try:
        with pymupdf.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text("text").strip()
                if page_text:
                    text += f"\n--- Halaman {page_num} ---\n{page_text}"
                else:
                    if reader:
                        pix = page.get_pixmap()
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        img_np = np.array(img)
                        ocr_result = reader.readtext(img_np, detail=0)
                        ocr_text = "\n".join(ocr_result)
                        text += f"\n--- Halaman {page_num} (OCR) ---\n{ocr_text}"
                    else:
                        text += f"\n--- Halaman {page_num} (Tidak ada teks dan OCR tidak tersedia) ---"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Memecah teks panjang menjadi potongan-potongan yang lebih kecil dan tumpang tindih.
    """
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

@document_bp.route('/process-assesmen', methods=['POST'])
@jwt_required()
@role_required("admin", "ketim")
def process_assesmen():
    """
    Endpoint API untuk memproses file PDF yang diunggah dan menyimpannya berdasarkan jenis file.
    Hasilnya disimpan ke database dan file FAISS yang sesuai.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file dalam permintaan'}), 400

    file = request.files['file']
    file_type = "assesmen"
    

    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
    
    if file_type not in ['assesmen', 'hasil lab']:
        return jsonify({'error': 'Jenis file tidak valid. Gunakan "assesmen" atau "hasil lab"'}), 400

    if not model or not reader:
        return jsonify({'error': 'Model tidak diinisialisasi. Silakan periksa log server.'}), 500


  
    temp_dir = "app/faisses/assesment"
 
    os.makedirs(temp_dir, exist_ok=True)
    temp_filepath = os.path.join(temp_dir, file.filename)
    faiss_file_path = os.path.join(temp_dir, f"{file_type}.faiss")
    print(f"Temporary file path: {temp_filepath}")

    try:
        file.save(temp_filepath)
        extracted_text = extract_text_from_pdf(temp_filepath)
        if extracted_text is None:
            return jsonify({'error': 'Gagal mengekstrak teks dari PDF'}), 500

        chunks = chunk_text(extracted_text)
        if not chunks:
            return jsonify({'message': 'Tidak ada teks yang dapat diproses dalam PDF.'}), 200
        
        emmendings = model.encode(chunks)
        index = initialize_faiss_index(faiss_file_path)
        index.add_with_ids(np.array(emmendings).astype('float32'), np.arange(len(emmendings)))
        save_faiss_index(index, faiss_file_path)

        return jsonify({'message': f'PDF diproses dan {len(chunks)} embeddings disimpan sebagai {file_type} di database dan {faiss_file_path}.'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        # if os.path.exists(temp_dir):
        #     os.rmdir(temp_dir)
@document_bp.route('/process-lab', methods=['POST'])
@jwt_required()
@role_required("admin", "user")
def process_lab():
    """
    Endpoint API untuk memproses file PDF yang diunggah dan menyimpannya berdasarkan jenis file.
    Hasilnya disimpan ke database dan file FAISS yang sesuai.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file dalam permintaan'}), 400

    file = request.files['file']
    cppt_id = request.form.get('cppt_id')
    file_type = "hasil lab"

    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

    if not model or not reader:
        return jsonify({'error': 'Model tidak diinisialisasi. Silakan periksa log server.'}), 500


    temp_dir = "app/faisses/lab_results"
    os.makedirs(f"{temp_dir}/{cppt_id}", exist_ok=True)
    temp_filepath = os.path.join(temp_dir, file.filename)
    faiss_file_path = os.path.join(f"{temp_dir}/{cppt_id}", f"{file_type}.faiss")

    try:
        file.save(temp_filepath)
        extracted_text = extract_text_from_pdf(temp_filepath)
        if extracted_text is None:
            return jsonify({'error': 'Gagal mengekstrak teks dari PDF'}), 500

        chunks = chunk_text(extracted_text)
        if not chunks:
            return jsonify({'message': 'Tidak ada teks yang dapat diproses dalam PDF.'}), 200
        
        emmendings = model.encode(chunks)
        index = initialize_faiss_index(faiss_file_path)
        index.add_with_ids(np.array(emmendings).astype('float32'), np.arange(len(emmendings)))
        save_faiss_index(index, faiss_file_path)

        return jsonify({'message': f'PDF diproses dan {len(chunks)} embeddings disimpan sebagai {file_type} di database dan {faiss_file_path}.'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        # if os.path.exists(temp_dir):
        #     os.rmdir(temp_dir)


@document_bp.route('/process-permenkes', methods=['POST'])
@jwt_required()
@role_required("admin", "ketim")
def process_permenkes():
    """
    Endpoint API untuk memproses file PDF yang diunggah dan menyimpannya berdasarkan jenis file.
    Hasilnya disimpan ke database dan file FAISS yang sesuai.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file dalam permintaan'}), 400

    file = request.files['file']
    file_type = "permenkes"

    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

    if not model or not reader:
        return jsonify({'error': 'Model tidak diinisialisasi. Silakan periksa log server.'}), 500


    temp_dir = "app/faisses/permenkes"
    os.makedirs(f"{temp_dir}", exist_ok=True)
    temp_filepath = os.path.join(temp_dir, file.filename)
    faiss_file_path = os.path.join(f"{temp_dir}", f"{file_type}.faiss")

    try:
        file.save(temp_filepath)
        extracted_text = extract_text_from_pdf(temp_filepath)
        if extracted_text is None:
            return jsonify({'error': 'Gagal mengekstrak teks dari PDF'}), 500

        chunks = chunk_text(extracted_text)
        if not chunks:
            return jsonify({'message': 'Tidak ada teks yang dapat diproses dalam PDF.'}), 200
        
        emmendings = model.encode(chunks)
        index = initialize_faiss_index(faiss_file_path)
        index.add_with_ids(np.array(emmendings).astype('float32'), np.arange(len(emmendings)))
        save_faiss_index(index, faiss_file_path)

        return jsonify({'message': f'PDF diproses dan {len(chunks)} embeddings disimpan sebagai {file_type} di database dan {faiss_file_path}.'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        # if os.path.exists(temp_dir):
        #     os.rmdir(temp_dir)


@document_bp.route('/process-siki-slki-sdki', methods=['POST'])
@jwt_required()
@role_required("admin", "ketim")
def process_siki_slki_sdki():
    """
    Endpoint API untuk memproses file PDF yang diunggah dan menyimpannya berdasarkan jenis file.
    Hasilnya disimpan ke database dan file FAISS yang sesuai.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file dalam permintaan'}), 400

    file = request.files['file']
    file_type = "siki-slki-sdki"

    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

    if not model or not reader:
        return jsonify({'error': 'Model tidak diinisialisasi. Silakan periksa log server.'}), 500


    temp_dir = "app/faisses/siki-slki-sdki"
    os.makedirs(f"{temp_dir}", exist_ok=True)
    temp_filepath = os.path.join(temp_dir, file.filename)
    faiss_file_path = os.path.join(f"{temp_dir}", f"{file_type}.faiss")

    try:
        file.save(temp_filepath)
        extracted_text = extract_text_from_pdf(temp_filepath)
        if extracted_text is None:
            return jsonify({'error': 'Gagal mengekstrak teks dari PDF'}), 500

        chunks = chunk_text(extracted_text)
        if not chunks:
            return jsonify({'message': 'Tidak ada teks yang dapat diproses dalam PDF.'}), 200
        
        emmendings = model.encode(chunks)
        index = initialize_faiss_index(faiss_file_path)
        index.add_with_ids(np.array(emmendings).astype('float32'), np.arange(len(emmendings)))
        save_faiss_index(index, faiss_file_path)

        return jsonify({'message': f'PDF diproses dan {len(chunks)} embeddings disimpan sebagai {file_type} di database dan {faiss_file_path}.'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        # if os.path.exists(temp_dir):
        #     os.rmdir(temp_dir)