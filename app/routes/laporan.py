from flask import Blueprint, request, jsonify
from app.model import db, Laporan, Patient, User
import os, json, pickle, re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
from flask_jwt_extended import jwt_required, get_jwt_identity
import numpy as np

# --- Load API Key ---
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY_KU")
api_model = os.getenv("API_MODEL")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# Inisialisasi Model Embedding
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Konfigurasi Path
FAISS_INDEX_FILE = "app/faisses/siki-slki-sdki/siki-slki-sdki.faiss"
MAPPING_FILE = "app/faisses/siki-slki-sdki/siki-slki-sdki.pkl"
BASE_DIR = "app/faisses/siki-slki-sdki"

laporan_bp = Blueprint("laporan_bp", __name__, url_prefix="/laporan")

# ==========================================================
# 1. HELPER FUNCTIONS (PRE-PROCESSING & SEARCH)
# ==========================================================

def safe_json(val):
    try:
        return json.loads(val) if val else []
    except:
        return []

def laporan_to_dict(l):
    return {
        "id": l.id,
        "patient_id": l.patient_id,
        "tanggal": l.tanggal.isoformat(),
        "subjective": l.subjective,
        "objective": l.objective,
        "assessment": l.assessment,
        "plan": l.plan,
        "tindakan_lanjutan": l.tindakan_lanjutan,
        "keterangan": l.keterangan,
        "SDKI": safe_json(l.SDKI),
        "SIKI": safe_json(l.SIKI),
        "SLKI": safe_json(l.SLKI),
        "user_id": l.user_id
    }

def load_dual_indexes(base_dir):
    """Load symptom + full FAISS index dan mapping"""
    symptom_faiss = os.path.join(base_dir, "symptom.faiss")
    symptom_pkl = os.path.join(base_dir, "symptom.pkl")
    full_faiss = os.path.join(base_dir, "full.faiss")
    full_pkl = os.path.join(base_dir, "full.pkl")

    if not all(map(os.path.exists, [symptom_faiss, symptom_pkl, full_faiss, full_pkl])):
        raise FileNotFoundError("Dual index belum tersedia. Upload PDF SDKI-SIKI-SLKI dulu.")

    symptom_index = faiss.read_index(symptom_faiss)
    full_index = faiss.read_index(full_faiss)

    with open(symptom_pkl, "rb") as f:
        symptom_map = pickle.load(f)

    with open(full_pkl, "rb") as f:
        full_map = pickle.load(f)

    return symptom_index, symptom_map, full_index, full_map

def structure_query_with_llm(raw_query):
    """
    Mengubah input mentah menjadi format Subjective/Objective 
    agar cocok dengan format index FAISS.
    """
    system_prompt = """
    Anda adalah asisten triase medis. 
    Tugas: Ekstrak input user menjadi dua bagian: DATA SUBJEKTIF dan DATA OBJEKTIF.
    
    Format Output Wajib (Jangan tambah kata lain):
    KELUHAN PASIEN (Subjektif): [Isi data subjektif, jika tidak ada tulis '-']
    TEMUAN KLINIS (Objektif): [Isi data objektif, jika tidak ada tulis '-']
    """
    
    try:
        completion = client.chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_query}
            ],
            temperature=0.1
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error structuring query: {e}")
        return f"KELUHAN PASIEN (Subjektif): {raw_query}\nTEMUAN KLINIS (Objektif): -"

def search_dual_index(query, symptom_index, symptom_map, full_map, k=3):
    """
    Melakukan pencarian dengan query terstruktur dan mengambil Top-K hasil.
    Mengembalikan: Full Context (gabungan), Symptom Context, dan Structured Query.
    """
    
    structured_query = structure_query_with_llm(query)
    print(f"DEBUG: Query Terstruktur -> \n{structured_query}")

    # 2. Encode & Search
    q_emb = model.encode([structured_query]).astype("float32")
    
    # Ambil k hasil teratas (misal 3)
    D, I = symptom_index.search(q_emb, k)

    if len(I[0]) == 0 or I[0][0] == -1:
        return None, None, structured_query

    combined_full_context = []
    combined_symptoms = []
    found_nos = set() # Untuk mencegah duplikasi diagnosa

    # 3. Looping Hasil FAISS (FIX LOGIKA LAMA)
    for idx in I[0]:
        idx = int(idx)
        if idx == -1: continue
        if idx not in symptom_map: continue
            
        candidate = symptom_map[idx]
        candidate_no = candidate["no"]
        
        # Skip jika nomor diagnosa ini sudah diambil
        if candidate_no in found_nos:
            continue
        found_nos.add(candidate_no)

        # Ambil Teks Gejala untuk referensi prompt
        combined_symptoms.append(f"[Kandidat Diagnosa NO {candidate_no}]:\n{candidate['text']}")

        # Ambil Full Text (SDKI, SIKI, SLKI) dari full_map
        for v in full_map.values():
            if v["no"] == candidate_no:
                header = f"--- OPSI DIAGNOSA KE-{len(found_nos)} (ID BUKU: {candidate_no}) ---"
                combined_full_context.append(f"{header}\n{v['text']}")
                break
    
    # Jika tidak ada hasil valid
    if not combined_full_context:
        return None, None, structured_query

    # Gabungkan string
    final_context = "\n\n".join(combined_full_context)
    final_symptoms = "\n\n".join(combined_symptoms)

    return final_context, final_symptoms, structured_query


# ==========================================================
# 2. ROUTE: CREATE LAPORAN (UPDATED)
# ==========================================================

@laporan_bp.route("/", methods=["POST"])
@jwt_required()
def create_laporan():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    payload = request.get_json()

    if not payload or not payload.get("query") or not payload.get("patient_id"):
        return jsonify({"status": 400, "message": "Harap isi query dan patient_id"}), 400

    query_text = payload["query"]
    patient_id = payload["patient_id"]
    patient = Patient.query.get(patient_id)

    if not patient:
        return jsonify({"status": 404, "message": "Pasien tidak ditemukan"}), 404
    if user.role.name != 'admin' and user.ruangan and patient.ruangan != user.ruangan:
        return jsonify({"status": 403, "message": "Akses ruangan ditolak"}), 403

    # --- 1. RETRIEVAL ---
    context_text = ""
    ai_subjective = "-"
    ai_objective = "-"

    try:
        symptom_index, symptom_map, full_index, full_map = load_dual_indexes(BASE_DIR)
        
        # Ambil 4 Kandidat
        res_full, _, res_structured = search_dual_index(
            query_text, symptom_index, symptom_map, full_map, k=4
        )

        if res_full:
            context_text = res_full
            # Parsing S/O dari AI Structure
            try:
                lines = res_structured.split('\n')
                for line in lines:
                    if "Subjektif" in line: ai_subjective = line.split(":", 1)[1].strip()
                    elif "Objektif" in line: ai_objective = line.split(":", 1)[1].strip()
            except: pass
        else:
            context_text = "DATA TIDAK DITEMUKAN."

    except Exception as e:
        print(f"Error: {e}")
        context_text = "Error retrieval."

    final_subjective = ai_subjective if ai_subjective != "-" and ai_subjective != "" else query_text
    final_objective = ai_objective if ai_objective != "-" else ""

    # --- 2. SYSTEM PROMPT (PERBAIKAN UTAMA: STRICT COPY PASTE) ---
    system_prompt = """
Anda adalah Sistem Otomasi CPPT (Catatan Perkembangan Pasien Terintegrasi).

TUGAS ANDA:
1. Baca "KONDISI PASIEN".
2. Pilih Semua "OPSI DIAGNOSA" yang relevan dari database yang paling cocok (1-4).
3. OUTPUT JSON HARUS BERISI DATA LENGKAP DARI OPSI YANG DIPILIH.
4. Gabungkan semua isi diagnosis yang dipilih menjadi SATU output JSON tunggal.

ATURAN KRUSIAL (JANGAN DILANGGAR):
- **SDKI, SIKI, SLKI**: Anda DILARANG MERINGKAS, MEMOTONG, atau MEMILIH POIN TERTENTU.
- **COPY-PASTE TOTAL**: Jika Anda memilih "Opsi Diagnosa 1", maka SELURUH TEKS intervensi, observasi, terapeutik, edukasi, dan kolaborasi yang ada di teks Opsi 1 harus dimasukkan ke dalam array JSON.
- **JANGAN ADA YANG TERTINGGAL**: Jika di teks asli ada 10 poin intervensi, di JSON output harus ada 10 string.
- **Subjective & Objective**: Isi sesuai input kondisi pasien.
- **Assessment**: Isi dengan Judul Diagnosa yang Anda pilih.

ATURAN PENGISIAN FIELD JSON:
1. **subjective & objective**: Isi dengan KONDISI PASIEN (Real).
2. **assessment**: Isi dengan JUDUL/NAMA DIAGNOSA yang dipilih (Contoh: "Nyeri Akut (D.0077)").
3. **SDKI**: Isi array ini HANYA dengan Nama Diagnosa, Kode, dan Penyebab/Faktor Risiko (jika ada di teks). **PENTING: JANGAN masukkan list "Data Subjektif" atau "Data Objektif" dari buku ke dalam array SDKI ini, karena itu akan duplikat dengan data pasien.**
4. **SIKI**: Salin SEMUA poin intervensi (Observasi, Terapeutik, Edukasi, Kolaborasi).
5. **SLKI**: Salin SEMUA kriteria hasil luaran.

ATURAN PEMILIHAN DIAGNOSA (CRITICAL):
1. Bandingkan "KONDISI PASIEN" dengan "DATA SUBJEKTIF/OBJEKTIF" pada setiap Opsi Diagnosa.
2. JANGAN MEMILIH Opsi Diagnosa yang data objektifnya TIDAK COCOK, meskipun ada satu atau dua kata kunci yang sama.
3. CONTOH: Jika pasien "Mulut Kering" tapi tidak ada luka, JANGAN PILIH "Gangguan Integritas Kulit" walaupun di database ada kata "kering". Pilih yang lebih relevan seperti "Defisit Nutrisi" atau "Hipovolemia" jika ada.
4. Prioritaskan diagnosa yang mencakup keluhan UTAMA pasien (misal: Kelemahan tubuh sesisi).

ATURAN PENGGABUNGAN DAN FORMATTING:
- **Assessment**: Gabungkan Judul Diagnosis dengan tanda koma atau simbol "&" (Contoh: "Nyeri Akut (D.0077) & Gangguan Mobilitas Fisik (D.0054)").
- **SDKI, SIKI, SLKI**: Masukkan isi dari SEMUA diagnosis yang dipilih ke dalam masing-masing array.
- **SEPARATOR (PENTING)**: Di dalam array SDKI, SIKI, dan SLKI, gunakan string khusus "--------------------" (garis putus-putus) untuk memisahkan item milik Diagnosis 1 dan Diagnosis 2.
- **Urutan**: Tulis item Diagnosis 1, lalu separator, lalu item Diagnosis 2.

CONTOH FORMAT LIST (Misalnya SIKI):
[
  "Observasi: Identifikasi skala nyeri",
  "Terapeutik: Berikan teknik relaksasi",
  "--------------------",  <-- INI SEPARATOR
  "Observasi: Identifikasi kekuatan otot (Milik Diagnosa 2)",
  "Terapeutik: Fasilitasi mobilisasi (Milik Diagnosa 2)"
]

ATURAN KONTEN:
- Tetap lakukan COPY-PASTE sesuai teks asli di database.
- Jangan meringkas kalimat.

FORMAT JSON OUTPUT:
{
  "subjective": "...",
  "objective": "...",
  "assessment": "JUDUL DIAGNOSA (Kode)",
  "plan": "Tulis 'Lakukan Intervensi Sesuai SIKI/SLKI'",
  "tindakan_lanjutan": "...",
  "keterangan": "...",
  "SDKI": ["Salin semua poin SDKI disini..."],
  "SIKI": ["Salin semua poin SIKI (Observasi, Terapeutik, Edukasi, Kolaborasi) disini..."],
  "SLKI": ["Salin semua kriteria hasil SLKI disini..."]
}
"""

    user_prompt_content = f"""
--- KONDISI PASIEN ---
Subjektif: {final_subjective}
Objektif: {final_objective}

--- PILIHAN DATA DARI DATABASE (OPSI LENGKAP) ---
{context_text}

--- INSTRUKSI ---
Pilih diagnosa yang paling relevan.
Lalu SALIN SEMUA ISINYA ke dalam JSON tanpa pengurangan sedikitpun.
"""

    # --- 3. EXECUTE LLM ---
    try:
        completion = client.chat.completions.create(
            model=api_model,
            temperature=0.1, # Suhu sangat rendah agar patuh copy-paste
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_content}
            ],
        )
        ai_json = completion.choices[0].message.content
        ai_json = re.sub(r"^```json\s*|\s*```$", "", ai_json.strip(), flags=re.MULTILINE)
        parsed = json.loads(ai_json)

    except Exception as e:
        parsed = {
            "subjective": final_subjective, "objective": final_objective,
            "assessment": "Gagal", "plan": "-", "tindakan_lanjutan": "-", "keterangan": str(e),
            "SDKI": [], "SIKI": [], "SLKI": []
        }

    # --- 4. SAVE TO DB ---
    # Validasi Assessment
    assess_str = parsed.get("assessment", "")
    if not assess_str or assess_str == "-":
        assess_str = "Diagnosa Keperawatan"

    laporan = Laporan(
        patient=patient,
        user_id=user_id,
        tanggal=datetime.utcnow(),
        subjective=parsed.get("subjective", "-"),
        objective=parsed.get("objective", "-"),
        assessment=assess_str,
        plan=parsed.get("plan", "-"),
        tindakan_lanjutan=parsed.get("tindakan_lanjutan", "-"),
        keterangan=parsed.get("keterangan", ""),
        SDKI=json.dumps(parsed.get("SDKI", [])),
        SIKI=json.dumps(parsed.get("SIKI", [])),
        SLKI=json.dumps(parsed.get("SLKI", []))
    )

    db.session.add(laporan)
    db.session.commit()

    return jsonify({
        "status": 201,
        "message": "Laporan berhasil dibuat (Full Standard Copy)",
        "data": laporan_to_dict(laporan)
    }), 201

@laporan_bp.route("/", methods=["GET"])
@jwt_required()
def get_laporans():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    query = Laporan.query
    if user.role.name != 'admin' and user.ruangan:
        query = query.join(Patient).filter(Patient.ruangan == user.ruangan)
    laporans = query.order_by(Laporan.tanggal.desc()).all()
    data = [laporan_to_dict(l) for l in laporans]
    return jsonify({"status": 200, "message": "Success", "data": data}), 200

@laporan_bp.route("/<int:id>", methods=["GET"])
@jwt_required()
def get_laporan(id):
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)

    laporan = Laporan.query.get(id)
    if not laporan:
        return jsonify({"status": 404, "message": "Laporan not found", "data": None}), 404

    patient = laporan.patient

    if user.role.name != 'admin' and user.ruangan and patient.ruangan != user.ruangan:
        return jsonify({"status": 403, "message": "Akses ditolak"}), 403

    return jsonify({
        "status": 200,
        "message": "success",
        "data": laporan_to_dict(laporan)
    }), 200

# ================== UPDATE ==================
@laporan_bp.route("/<int:id>", methods=["PUT"])
@jwt_required()
def update_laporan(id):
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)

    laporan = Laporan.query.get(id)
    if not laporan:
        return jsonify({"status": 404, "message": "Laporan not found", "data": None}), 404

    patient = laporan.patient
    if user.role.name != 'admin' and user.ruangan and patient.ruangan != user.ruangan:
        return jsonify({"status": 403, "message": "Akses ditolak"}), 403

    payload = request.get_json()

    for field in [
        "subjective",
        "objective",
        "assessment",
        "plan",
        "tindakan_lanjutan",
        "keterangan"
    ]:
        if field in payload:
            setattr(laporan, field, payload[field])

    for field in ["SDKI", "SIKI", "SLKI"]:
        if field in payload:
            setattr(laporan, field, json.dumps(payload[field]))

    db.session.commit()

    return jsonify({
        "status": 200,
        "message": "Laporan updated successfully",
        "data": laporan_to_dict(laporan)
    }), 200

# ================== DELETE ==================
@laporan_bp.route("/<int:id>", methods=["DELETE"])
@jwt_required()
def delete_laporan(id):
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)

    laporan = Laporan.query.get(id)
    if not laporan:
        return jsonify({"status": 404, "message": "Laporan not found", "data": None}), 404

    patient = laporan.patient
    if user.role.name != 'admin' and user.ruangan and patient.ruangan != user.ruangan:
        return jsonify({"status": 403, "message": "Akses ditolak"}), 403

    db.session.delete(laporan)
    db.session.commit()

    return jsonify({
        "status": 200,
        "message": "Laporan deleted successfully",
        "data": None
    }), 200

# ================== SEARCH ==================
@laporan_bp.route("/search", methods=["GET"])
@jwt_required()
def search_laporan():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)

    keyword = request.args.get("q")
    if not keyword:
        return jsonify({"status": 400, "message": "Query parameter 'q' required", "data": None}), 400

    query = Laporan.query.filter(
        (Laporan.subjective.ilike(f"%{keyword}%")) |
        (Laporan.objective.ilike(f"%{keyword}%")) |
        (Laporan.assessment.ilike(f"%{keyword}%")) |
        (Laporan.plan.ilike(f"%{keyword}%")) |
        (Laporan.keterangan.ilike(f"%{keyword}%"))
    )

    if user.role.name != 'admin' and user.ruangan:
        query = query.join(Patient).filter(Patient.ruangan == user.ruangan)

    results = query.all()

    return jsonify({
        "status": 200,
        "message": "success",
        "data": [laporan_to_dict(l) for l in results]
    }), 200
