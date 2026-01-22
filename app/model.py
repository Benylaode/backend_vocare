from . import db
import enum
from werkzeug.security import generate_password_hash, check_password_hash
# HAPUS IMPORT JSONB DARI DIALECT POSTGRESQL
# GANTI DENGAN IMPORT JSON DARI SQLALCHEMY UTAMA
from sqlalchemy import JSON, Enum 
from datetime import datetime, date

# Tabel Asosiasi Pasien - User
patient_user = db.Table(
    "patient_user",
    db.Column("patient_id", db.Integer, db.ForeignKey("patients.id"), primary_key=True),
    db.Column("user_id", db.Integer, db.ForeignKey("users.id"), primary_key=True)
)

class RoleEnum(enum.Enum):
    admin = "admin"
    user = "user"
    ketim = "ketim"

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    # Gunakan db.Enum(RoleEnum) agar kompatibel
    role = db.Column(db.Enum(RoleEnum), nullable=True) 
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.Text, nullable=False)
    ruangan = db.Column(db.String(50), nullable=True) 

    patients = db.relationship("Patient", secondary=patient_user, back_populates="users")
    
    assesments = db.relationship("Assesment", back_populates="user", cascade="all, delete-orphan")
    laporans = db.relationship("Laporan", back_populates="user", cascade="all, delete-orphan")
    cppts = db.relationship("CPPT", back_populates="user", cascade="all, delete-orphan")
    intervensis = db.relationship("Intervensi", back_populates="user", cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Patient(db.Model):
    __tablename__ = "patients"
    id = db.Column(db.Integer, primary_key=True)
    no_rekam_medis = db.Column(db.String(50), unique=True, nullable=True)
    ruangan = db.Column(db.String(50), nullable=True)
    nama = db.Column(db.String(120), nullable=False)
    tgl_lahir = db.Column(db.Date, nullable=True)
    jenis_kelamin = db.Column(db.String(10), nullable=True)
    
    alamat = db.Column(db.Text, nullable=True)
    agama = db.Column(db.String(50), nullable=True)
    pekerjaan = db.Column(db.String(120), nullable=True)
    status_perkawinan = db.Column(db.String(50), nullable=True)
    penanggung_jawab = db.Column(db.String(120), nullable=True)
    hubungan_penanggung_jawab = db.Column(db.String(50), nullable=True)
    kontak_penanggung_jawab = db.Column(db.String(50), nullable=True)

    status_rawat = db.Column(
        db.Enum("rawat_inap", "rawat_jalan", "pulang", "meninggal", name="status_rawat_enum"),
        nullable=False,
        default="rawat_inap"
    )

    users = db.relationship("User", secondary=patient_user, back_populates="patients")
    assesments = db.relationship("Assesment", back_populates="patient", cascade="all, delete-orphan")
    laporans = db.relationship("Laporan", back_populates="patient", cascade="all, delete-orphan")
    cppts = db.relationship("CPPT", back_populates="patient", cascade="all, delete-orphan")
    intervensis = db.relationship("Intervensi", back_populates="patient", cascade="all, delete-orphan")

    def umur(self):
        if self.tgl_lahir:
            today = date.today()
            return today.year - self.tgl_lahir.year - (
                (today.month, today.day) < (self.tgl_lahir.month, self.tgl_lahir.day)
            )
        return None

class Assesment(db.Model):
    __tablename__ = "assesments"
    id = db.Column(db.Integer, primary_key=True)
    tanggal = db.Column(db.DateTime, default=datetime.utcnow)
    
    # --- PERBAIKAN: Ganti JSONB (Postgres only) ke JSON (Generic) ---
    data = db.Column(JSON, nullable=False, default={}) 
    
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    
    patient = db.relationship("Patient", back_populates="assesments")
    user = db.relationship("User", back_populates="assesments")
    laporans = db.relationship("Laporan", back_populates="assesment", cascade="all, delete-orphan")

class Laporan(db.Model):
    __tablename__ = "laporan"
    id = db.Column(db.Integer, primary_key=True)
    tanggal = db.Column(db.DateTime, default=datetime.utcnow)

    subjective = db.Column(db.Text, nullable=True)
    objective = db.Column(db.Text, nullable=True)
    assessment = db.Column(db.Text, nullable=True)
    plan = db.Column(db.Text, nullable=True)
    
    SDKI = db.Column(db.Text, nullable=True) 
    SIKI = db.Column(db.Text, nullable=True) 
    SLKI = db.Column(db.Text, nullable=True) 
    
    tindakan_lanjutan = db.Column(db.Text, nullable=True)
    keterangan = db.Column(db.Text, nullable=True)
    dokter = db.Column(db.String(120), nullable=True)
    signature = db.Column(db.Text, nullable=True)

    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    assesment_id = db.Column(db.Integer, db.ForeignKey("assesments.id"), nullable=True)

    patient = db.relationship("Patient", back_populates="laporans")
    user = db.relationship("User", back_populates="laporans")
    assesment = db.relationship("Assesment", back_populates="laporans")
    cppts = db.relationship("CPPT", back_populates="laporan")

class CPPT(db.Model):
    __tablename__ = "CPPT"
    id = db.Column(db.Integer, primary_key=True)
    tanggal = db.Column(db.DateTime, default=datetime.utcnow)
    shift = db.Column(db.String(20), nullable=True)
    jabatan = db.Column(db.String(120), nullable=True)

    subjective = db.Column(db.Text, nullable=True)
    objective = db.Column(db.Text, nullable=True)
    assessment = db.Column(db.Text, nullable=True)
    plan = db.Column(db.Text, nullable=True)
    keterangan = db.Column(db.Text, nullable=True)
    dokter = db.Column(db.String(120), nullable=True)
    signature = db.Column(db.Text, nullable=True)

    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    laporan_id = db.Column(db.Integer, db.ForeignKey("laporan.id"), nullable=True)

    patient = db.relationship("Patient", back_populates="cppts")
    user = db.relationship("User", back_populates="cppts")
    laporan = db.relationship("Laporan", back_populates="cppts")

class Intervensi(db.Model):
    __tablename__ = "intervensi"
    id = db.Column(db.Integer, primary_key=True)
    tanggal = db.Column(db.DateTime, default=datetime.utcnow)
    implementasi = db.Column(db.Text, nullable=True)
    evaluasi = db.Column(db.Text, nullable=True)

    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    
    patient = db.relationship("Patient", back_populates="intervensis")
    user = db.relationship("User", back_populates="intervensis")