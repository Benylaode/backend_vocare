from . import db
import enum
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.dialects.postgresql import ARRAY, FLOAT
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
from datetime import date
from sqlalchemy import Enum

patient_user = db.Table(
    "patient_user",
    db.Column("patient_id", db.Integer, db.ForeignKey("patients.id"), primary_key=True),
    db.Column("user_id", db.Integer, db.ForeignKey("users.id"), primary_key=True)
)

class RoleEnum(enum.Enum):
    admin = "admin"
    user = "user"
    editor = "ketim"

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    role = db.Column(db.Enum(RoleEnum), default=RoleEnum.user, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.Text, nullable=False)
    patients = db.relationship("Patient", secondary=patient_user, back_populates="users")
    cppt = db.relationship("CPPT", back_populates="user")
    laporan = db.relationship("Laporan", back_populates="user")
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
class Assesment(db.Model):
    __tablename__ = "assesments"
    id = db.Column(db.Integer, primary_key=True)
    tanggal = db.Column(db.DateTime, default=datetime.utcnow)
    perawat = db.Column(db.String(120), nullable=True)

    data = db.Column(JSONB, nullable=False, default={})
    patient = db.relationship(
        "Patient",
        back_populates="assesment",
        uselist=False,
        cascade="all, delete-orphan"
    )
    
class CPPT(db.Model):
    __tablename__ = "CPPT"
    id = db.Column(db.Integer, primary_key=True)

    tanggal = db.Column(db.DateTime, default=datetime.utcnow)
    jabatan = db.Column(db.String(120), nullable=True)  

    subjective = db.Column(db.Text, nullable=True)
    objective = db.Column(db.Text, nullable=True)
    assessment = db.Column(db.Text, nullable=True)
    plan = db.Column(db.Text, nullable=True)
    keterangan = db.Column(db.Text, nullable=True)

    dokter = db.Column(db.String(120), nullable=True)
    signature = db.Column(db.Text, nullable=True)

    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False)
    patient = db.relationship("Patient", back_populates="cppt")
    
    laporan = db.relationship("Laporan", back_populates="cppt", uselist=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    user = db.relationship("User", back_populates="cppt")

class Laporan(db.Model):
    __tablename__ = "laporan"

    id = db.Column(db.Integer, primary_key=True)
    tanggal = db.Column(db.DateTime, default=datetime.utcnow) 

    subjective = db.Column(db.Text, nullable=True)
    objective = db.Column(db.Text, nullable=True)
    assessment = db.Column(db.Text, nullable=True)
    plan = db.Column(db.Text, nullable=True)
    keterangan = db.Column(db.Text, nullable=True)

    dokter = db.Column(db.String(120), nullable=True)
    signature = db.Column(db.Text, nullable=True)

    tindakan_lanjutan = db.Column(db.Text, nullable=True)  
    SDKI = db.Column(db.Text, nullable=True)     
    SLKI = db.Column(db.Text, nullable=True)              
    SIKI = db.Column(db.Text, nullable=True)               

    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False)
    patient = db.relationship("Patient", back_populates="laporan")

    cppt = db.relationship("CPPT", back_populates="laporan")
    cppt_id = db.Column(db.Integer, db.ForeignKey("CPPT.id"), nullable=True)

    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    user = db.relationship("User", back_populates="laporan")

class Patient(db.Model):
    __tablename__ = "patients"

    id = db.Column(db.Integer, primary_key=True)
    no_rekam_medis = db.Column(db.String(50), unique=True, nullable=False)   
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

    assesment = db.relationship("Assesment", back_populates="patient")
    assesment_id = db.Column(db.Integer, db.ForeignKey("assesments.id"), nullable=True)
    users = db.relationship("User", secondary=patient_user, back_populates="patients")
    cppt = db.relationship("CPPT", back_populates="patient", cascade="all, delete-orphan")
    laporan = db.relationship("Laporan", back_populates="patient", cascade="all, delete-orphan")

    status_rawat = db.Column(
        Enum("rawat_inap", "rawat_jalan", name="status_rawat_enum"),
        nullable=False,
        default="rawat_jalan"
    )

    def umur(self):
        """Hitung umur pasien otomatis"""
        if self.tgl_lahir:
            today = date.today()
            return today.year - self.tgl_lahir.year - (
                (today.month, today.day) < (self.tgl_lahir.month, self.tgl_lahir.day)
            )
        return None
