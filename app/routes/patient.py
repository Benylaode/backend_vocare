from flask import Blueprint, request, jsonify
from app.model import db, Patient, Assesment
from datetime import datetime
from flask_jwt_extended import jwt_required
from app.utils import role_required

patient_bp = Blueprint("patient_bp", __name__, url_prefix="/patients")

@patient_bp.route("/", methods=["GET"])
def get_patients():
    patients = Patient.query.all()
    data = [
        {
            "id": p.id,
            "no_rekam_medis": p.no_rekam_medis,
            "id_assesment": p.assesment_id,
            "nama": p.nama,
            "tgl_lahir": p.tgl_lahir.isoformat() if p.tgl_lahir else None,
            "jenis_kelamin": p.jenis_kelamin,
            "alamat": p.alamat,
            "agama": p.agama,
            "pekerjaan": p.pekerjaan,
            "status_perkawinan": p.status_perkawinan,
            "penanggung_jawab": p.penanggung_jawab,
            "hubungan_penanggung_jawab": p.hubungan_penanggung_jawab,
            "kontak_penanggung_jawab": p.kontak_penanggung_jawab,
            "status_rawat": p.status_rawat,
        }
        for p in patients
    ]
    return jsonify({"status": 200, "message": "Success", "data": data}), 200

@patient_bp.route("/<int:patient_id>", methods=["GET"])
def get_patient(patient_id):
    p = Patient.query.get(patient_id)
    if not p:
        return jsonify({"status": 404, "message": "Patient not found", "data": None}), 404

    data = {
        "id": p.id,
        "no_rekam_medis": p.no_rekam_medis,
        "id_assesment": p.assesment_id,
        "nama": p.nama,
        "tgl_lahir": p.tgl_lahir.isoformat() if p.tgl_lahir else None,
        "jenis_kelamin": p.jenis_kelamin,
        "alamat": p.alamat,
        "agama": p.agama,
        "pekerjaan": p.pekerjaan,
        "status_perkawinan": p.status_perkawinan,
        "penanggung_jawab": p.penanggung_jawab,
        "hubungan_penanggung_jawab": p.hubungan_penanggung_jawab,
        "kontak_penanggung_jawab": p.kontak_penanggung_jawab,
        "status_rawat": p.status_rawat,
    }
    return jsonify({"status": 200, "message": "Success", "data": data}), 200


@patient_bp.route("/", methods=["POST"])
@jwt_required()
@role_required("admin", "user")
def create_patient():
    payload = request.get_json()

    if not payload or not payload.get("id_assesment") or not payload.get("nama"):
        return jsonify({"status": 400, "message": "Fields required: id_assesment, nama"}), 400

    id_assesment = payload["id_assesment"]
    nama = payload["nama"]

    assesment = Assesment.query.filter_by(id=id_assesment).first()
    if not assesment:
        return jsonify({"status": 404, "message": "Assesment not found"}), 404

    try:
        data = assesment.data
        if isinstance(data, str):
            import json
            if data.startswith("```json"):
                data = data[len("```json"):].strip()
            if data.endswith("```"):
                data = data[:-3].strip()
            data = json.loads(data)
    except Exception:
        return jsonify({"status": 500, "message": "Invalid JSON in assesment"}), 500

    if Patient.query.filter_by(assesment_id=id_assesment).first():
        return jsonify({"status": 400, "message": "Patient with this id_assesment already exists"}), 400

    info_umum = data.get("asesmen_awal_keperawatan", {}).get("informasi_umum", {})

    # ambil nama: bisa "nama" atau "nama_pasien"
    nama_pasien = info_umum.get("nama") or info_umum.get("nama_pasien") or nama
    no_rekam_medis = info_umum.get("no_rm") or info_umum.get("kode_rm") or nama

    # ambil tgl lahir (nama key di JSON: 'tanggal_lahir')
    tgl_lahir = info_umum.get("tanggal_lahir")
    if tgl_lahir and isinstance(tgl_lahir, str):
        try:
            from datetime import datetime
            tgl_lahir = datetime.strptime(tgl_lahir, "%d %B %Y").date()
        except Exception:
            tgl_lahir = None

    # ambil penanggung jawab (bisa string atau object)
    pj = info_umum.get("penanggung_jawab")
    if isinstance(pj, dict):
        nama_pj = pj.get("nama")
        hubungan_pj = pj.get("hubungan")
        kontak_pj = pj.get("kontak")
    else:
        nama_pj = pj
        hubungan_pj = info_umum.get("hubungan_penanggung_jawab")
        kontak_pj = info_umum.get("kontak_penanggung_jawab")

    new_patient = Patient(
        assesment_id=id_assesment,
        nama=nama_pasien,
        no_rekam_medis=no_rekam_medis,
        tgl_lahir=tgl_lahir,
        jenis_kelamin=info_umum.get("jenis_kelamin"),
        alamat=info_umum.get("alamat"),
        agama=info_umum.get("agama"),
        pekerjaan=info_umum.get("pekerjaan"),
        status_perkawinan=info_umum.get("status_perkawinan"),
        penanggung_jawab=nama_pj,
        hubungan_penanggung_jawab=hubungan_pj,
        kontak_penanggung_jawab=kontak_pj,
        status_rawat="rawat_inap"
    )

    db.session.add(new_patient)
    db.session.commit()

    return jsonify({
        "status": 201,
        "message": "Patient created from assesment",
        "data": {"id": new_patient.id}
    }), 201


@patient_bp.route("/<int:patient_id>", methods=["PUT"])
@jwt_required()
@role_required("admin", "user")
def update_patient(patient_id):
    p = Patient.query.get(patient_id)
    if not p:
        return jsonify({"status": 404, "message": "Patient not found", "data": None}), 404

    payload = request.get_json()
    if not payload:
        return jsonify({"status": 400, "message": "No data provided", "data": None}), 400

    for field in [
        "no_rekam_medis", "nama", "tgl_lahir", "jenis_kelamin", "alamat", "agama", "pekerjaan",
        "status_perkawinan", "penanggung_jawab", "hubungan_penanggung_jawab", "kontak_penanggung_jawab",
        "status_rawat"
    ]:
        if field in payload:
            if field == "tgl_lahir" and isinstance(payload[field], str):
                setattr(p, field, datetime.fromisoformat(payload[field]).date())
            else:
                setattr(p, field, payload[field])

    db.session.commit()
    return jsonify({"status": 200, "message": "Patient updated", "data": {"id": p.id}}), 200


@patient_bp.route("/<int:patient_id>", methods=["DELETE"])
@jwt_required()
@role_required("admin", "user")
def delete_patient(patient_id):
    p = Patient.query.get(patient_id)
    if not p:
        return jsonify({"status": 404, "message": "Patient not found", "data": None}), 404

    db.session.delete(p)
    db.session.commit()
    return jsonify({"status": 200, "message": "Patient deleted", "data": {"id": p.id}}), 200
