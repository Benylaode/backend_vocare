import pytest
import json
from unittest.mock import patch, MagicMock
from datetime import datetime, time

# ==========================================
# 1. SETUP MOCK DATETIME & DATA DUMMY
# ==========================================

# Kita buat class datetime palsu yang mewarisi datetime asli
# Ini kuncinya: agar saat kode memanggil .time(), hasilnya tetap valid
class MockDatetime(datetime):
    # Waktu default awal (bisa diubah-ubah di tengah test)
    _now = datetime(2026, 1, 23, 8, 0, 0) 

    @classmethod
    def utcnow(cls):
        return cls._now

# JSON Response Dummy untuk AI (agar test cepat & hemat kuota)
AI_ASSESMENT_RESPONSE = json.dumps({
    "informasi_umum": {"kesadaran": "Compos Mentis"},
    "tanda_vital": {"td": "120/80", "nadi": "80", "suhu": "36.5"},
    "keluhan_utama": "Nyeri dada sebelah kiri menjalar ke bahu",
    "rencana_asuhan_keperawatan": []
})

AI_LAPORAN_RESPONSE = json.dumps({
    "subjective": "Pasien mengeluh nyeri dada skala 6/10",
    "objective": "TD 120/80, Nadi 80, Wajah tampak meringis menahan sakit",
    "assessment": "Nyeri Akut b.d Agen Pencedera Fisiologis",
    "plan": "Lakukan manajemen nyeri, kolaborasi analgesik",
    "SDKI": ["Nyeri Akut"],
    "SIKI": ["Manajemen Nyeri", "Pemberian Analgesik"],
    "SLKI": ["Tingkat Nyeri Menurun"]
})

AI_CPPT_RESPONSE = json.dumps({
    "subjective": "Nyeri berkurang menjadi skala 3/10",
    "objective": "Pasien tampak lebih tenang, TD 110/70",
    "assessment": "Masalah Nyeri Akut teratasi sebagian",
    "plan": "Lanjutkan intervensi: teknik relaksasi",
    "keterangan": "Pasien kooperatif"
})

# ==========================================
# 2. SKENARIO TEST FULL JOURNEY
# ==========================================

@patch('app.routes.assesments.client.chat.completions.create')
@patch('app.routes.laporan.client.chat.completions.create')
@patch('app.routes.cppt.client.chat.completions.create')
# PERBAIKAN DI SINI: Gunakan 'new=' untuk mengganti class datetime sepenuhnya
@patch('app.routes.cppt.datetime', new=MockDatetime) 
def test_full_patient_care_cycle(mock_ai_cppt, mock_ai_laporan, mock_ai_assesment, client, perawat_mawar_token):
    """
    Simulasi Lengkap:
    1. Pasien Masuk -> 2. Asesmen -> 3. Laporan (Askeb) -> 
    4. Intervensi -> 5. CPPT (Pagi, Sore, Malam, Besok Pagi)
    """
    
    # Setup Token Auth
    headers = {"Authorization": f"Bearer {perawat_mawar_token}"}
    
    print("\n\n=== MEMULAI SIMULASI PERAWATAN PASIEN ===")

    # ---------------------------------------------------------
    # TAHAP 1: ADMISI PASIEN (Patient Admission)
    # ---------------------------------------------------------
    rm_number = "888001"
    patient_payload = {
        "nama": "Budi Santoso",
        "no_rekam_medis": rm_number,
        "ruangan": "Mawar",
        "tgl_lahir": "1980-05-20",
        "jenis_kelamin": "Laki-laki"
    }
    
    res_pat = client.post('/patients/', json=patient_payload, headers=headers)
    assert res_pat.status_code == 201
    patient_id = res_pat.json['data']['id']
    print(f"[1] ✅ Pasien Terdaftar: {patient_payload['nama']} (RM: {rm_number}, ID: {patient_id})")


    # ---------------------------------------------------------
    # TAHAP 2: ASESMEN AWAL (Initial Assessment)
    # ---------------------------------------------------------
    # Siapkan jawaban AI palsu
    mock_ai_assesment.return_value.choices = [MagicMock(message=MagicMock(content=AI_ASSESMENT_RESPONSE))]

    assesment_payload = {
        "patient_id": patient_id,
        "query": f"Pasien RM {rm_number} masuk dengan keluhan nyeri dada khas jantung."
    }
    res_ass = client.post('/assesments/', json=assesment_payload, headers=headers)
    assert res_ass.status_code == 201
    print(f"[2] ✅ Asesmen Awal Selesai. (ID Asesmen: {res_ass.json['data']['id']})")


    # ---------------------------------------------------------
    # TAHAP 3: PEMBUATAN LAPORAN/ASKEB (Care Plan)
    # ---------------------------------------------------------
    mock_ai_laporan.return_value.choices = [MagicMock(message=MagicMock(content=AI_LAPORAN_RESPONSE))]

    laporan_payload = {
        "patient_id": patient_id,
        "query": f"Buatkan askeb lengkap untuk pasien RM {rm_number}."
    }
    res_lap = client.post('/laporan/', json=laporan_payload, headers=headers)
    assert res_lap.status_code == 201
    
    diagnosa = res_lap.json['data']['SDKI']
    print(f"[3] ✅ Laporan/Askeb Terbentuk. Diagnosa Utama: {diagnosa}")


    # ---------------------------------------------------------
    # TAHAP 4: INTERVENSI & CPPT HARIAN (Daily Care Loop)
    # ---------------------------------------------------------
    
    # A. Intervensi Keperawatan
    intervensi_payload = {
        "patient_id": patient_id,
        "user_id": 1, # Asumsi ID dari token
        "implementasi": "Mengajarkan teknik relaksasi napas dalam",
        "evaluasi": "Pasien mampu mempraktikkan dengan baik"
    }
    res_int = client.post('/intervensi/', json=intervensi_payload, headers=headers)
    assert res_int.status_code == 201
    print(f"[4] ✅ Intervensi Dilakukan: {intervensi_payload['implementasi']}")

    # Setup AI untuk semua CPPT
    mock_ai_cppt.return_value.choices = [MagicMock(message=MagicMock(content=AI_CPPT_RESPONSE))]

    # B. CPPT Shift PAGI (Jam 08:00)
    MockDatetime._now = datetime(2026, 1, 23, 8, 0, 0) # Set Waktu
    
    res_pagi = client.post('/cppt/', json={"patient_id": patient_id, "query": "Evaluasi pagi"}, headers=headers)
    assert res_pagi.status_code == 201
    # Validasi output pesan mengandung kata 'Pagi'
    assert "Shift Pagi" in res_pagi.json['message'] 
    print(f"    -> 🌞 CPPT Shift PAGI tercatat.")

    # C. CPPT Shift SORE (Jam 16:00)
    MockDatetime._now = datetime(2026, 1, 23, 16, 0, 0) # Ubah Waktu
    
    res_sore = client.post('/cppt/', json={"patient_id": patient_id, "query": "Evaluasi sore"}, headers=headers)
    assert res_sore.status_code == 201
    assert "Shift Sore" in res_sore.json['message']
    print(f"    -> 🌤️  CPPT Shift SORE tercatat.")

    # D. CPPT Shift MALAM (Jam 22:00)
    MockDatetime._now = datetime(2026, 1, 23, 22, 0, 0) # Ubah Waktu
    
    res_malam = client.post('/cppt/', json={"patient_id": patient_id, "query": "Evaluasi malam"}, headers=headers)
    assert res_malam.status_code == 201
    assert "Shift Malam" in res_malam.json['message']
    print(f"    -> 🌙 CPPT Shift MALAM tercatat.")


    # ---------------------------------------------------------
    # TAHAP 5: HARI KEDUA (Follow Up Day 2)
    # ---------------------------------------------------------
    
    # Pindah ke hari berikutnya jam 07:30 pagi
    MockDatetime._now = datetime(2026, 1, 24, 7, 30, 0)
    
    res_d2 = client.post('/cppt/', json={"patient_id": patient_id, "query": "Pasien rencana pulang"}, headers=headers)
    assert res_d2.status_code == 201
    assert "Shift Pagi" in res_d2.json['message']
    print(f"[5] ✅ CPPT Hari Ke-2 (Pagi) Berhasil. Pasien siap pulang.")

    # Verifikasi total data CPPT tersimpan
    # (Opsional: memanggil GET untuk memastikan jumlah item)
    res_all_cppt = client.get('/cppt/', headers=headers)
    total_cppt_pasien = len([x for x in res_all_cppt.json['data'] if x['patient_id'] == patient_id])
    
    print(f"\n[INFO] Total CPPT Tersimpan untuk Pasien {rm_number}: {total_cppt_pasien} record.")
    assert total_cppt_pasien == 4 # 3 Shift hari pertama + 1 Shift hari kedua

    print("\n=== SEMUA SKENARIO BERHASIL DIJALANKAN TANPA ERROR ===")