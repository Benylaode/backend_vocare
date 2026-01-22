from flask import Blueprint, request, jsonify
from app.model import db, Patient, User
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime

patient_bp = Blueprint("patient_bp", __name__, url_prefix="/patients")
MAX_PATIENTS_PER_ROOM = 10 

# --- READ (Get All) ---
@patient_bp.route("/", methods=["GET"])
@jwt_required()
def get_patients():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    
    query = Patient.query
    if user.role.name != 'admin' and user.ruangan:
        query = query.filter_by(ruangan=user.ruangan)
    
    patients = query.order_by(Patient.id.desc()).all()
    data = [{
        "id": p.id, "no_rekam_medis": p.no_rekam_medis, "nama": p.nama,
        "ruangan": p.ruangan, "tgl_lahir": p.tgl_lahir.isoformat() if p.tgl_lahir else None,
        "umur": p.umur(), "jenis_kelamin": p.jenis_kelamin
    } for p in patients]
    
    return jsonify({"status": 200, "message": "Success", "data": data}), 200

# --- READ (Get One) ---
@patient_bp.route("/<int:patient_id>", methods=["GET"])
@jwt_required()
def get_patient_by_id(patient_id):
    patient = Patient.query.get(patient_id)
    if not patient:
        return jsonify({"status": 404, "message": "Pasien tidak ditemukan"}), 404
    
    return jsonify({
        "status": 200, "message": "Success",
        "data": {
            "id": patient.id, "no_rekam_medis": patient.no_rekam_medis,
            "nama": patient.nama, "ruangan": patient.ruangan,
            "tgl_lahir": patient.tgl_lahir.isoformat() if patient.tgl_lahir else None,
            "jenis_kelamin": patient.jenis_kelamin, "alamat": patient.alamat,
            "penanggung_jawab": patient.penanggung_jawab
        }
    }), 200

# --- CREATE ---
@patient_bp.route("/", methods=["POST"])
@jwt_required()
def create_patient():
    data = request.get_json()
    if not all(k in data for k in ["nama", "no_rekam_medis", "ruangan"]):
        return jsonify({"status": 400, "message": "Data wajib: nama, no_rekam_medis, ruangan"}), 400

    # Cek Kapasitas Ruangan
    count = Patient.query.filter_by(ruangan=data["ruangan"], status_rawat="rawat_inap").count()
    if count >= MAX_PATIENTS_PER_ROOM:
        return jsonify({"status": 400, "message": f"Ruangan {data['ruangan']} penuh (Max {MAX_PATIENTS_PER_ROOM})"}), 400

    if Patient.query.filter_by(no_rekam_medis=data["no_rekam_medis"]).first():
        return jsonify({"status": 409, "message": "No Rekam Medis sudah ada"}), 409
        
    try:
        tgl = datetime.strptime(data.get("tgl_lahir"), "%Y-%m-%d").date() if data.get("tgl_lahir") else None
    except: return jsonify({"status": 400, "message": "Format tanggal salah"}), 400

    p = Patient(
        nama=data["nama"], no_rekam_medis=data["no_rekam_medis"], ruangan=data["ruangan"],
        tgl_lahir=tgl, jenis_kelamin=data.get("jenis_kelamin"), alamat=data.get("alamat"),
        status_rawat="rawat_inap", penanggung_jawab=data.get("penanggung_jawab")
    )
    db.session.add(p)
    db.session.commit()
    
    return jsonify({"status": 201, "message": "Pasien terdaftar", "data": {"id": p.id}}), 201

# --- UPDATE ---
@patient_bp.route("/<int:patient_id>", methods=["PUT"])
@jwt_required()
def update_patient(patient_id):
    patient = Patient.query.get(patient_id)
    if not patient: return jsonify({"status": 404, "message": "Pasien tidak ditemukan"}), 404
    
    data = request.get_json()
    
    # Validasi Pindah Ruangan
    if "ruangan" in data and data["ruangan"] != patient.ruangan:
        count = Patient.query.filter_by(ruangan=data["ruangan"], status_rawat="rawat_inap").count()
        if count >= MAX_PATIENTS_PER_ROOM:
            return jsonify({"status": 400, "message": f"Ruangan tujuan penuh"}), 400
        patient.ruangan = data["ruangan"]

    if "nama" in data: patient.nama = data["nama"]
    if "alamat" in data: patient.alamat = data["alamat"]
    if "status_rawat" in data: patient.status_rawat = data["status_rawat"]
    if "penanggung_jawab" in data: patient.penanggung_jawab = data["penanggung_jawab"]
    
    if "tgl_lahir" in data:
        try: patient.tgl_lahir = datetime.strptime(data["tgl_lahir"], "%Y-%m-%d").date()
        except: return jsonify({"status": 400, "message": "Format tanggal salah"}), 400

    db.session.commit()
    return jsonify({"status": 200, "message": "Pasien diperbarui"}), 200

@patient_bp.route("/<int:patient_id>", methods=["DELETE"])
@jwt_required()
def delete_patient(patient_id):
    patient = Patient.query.get(patient_id)
    if not patient: return jsonify({"status": 404, "message": "Pasien tidak ditemukan"}), 404
    
    db.session.delete(patient)
    db.session.commit()
    return jsonify({"status": 200, "message": "Pasien dihapus"}), 200