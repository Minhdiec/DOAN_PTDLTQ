import json
import sys
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import ham_dung_chung as hdc


## 1. Đọc mô hình và dữ liệu

mo_hinh = joblib.load('../mo_hinh/mo_hinh.joblib')
mo_hinh_cam_xuc_absa = joblib.load('../mo_hinh/mo_hinh_cam_xuc_absa.joblib')
df = pd.read_csv('../dataset/du_lieu_dac_trung.csv')
df_absa = pd.read_csv('../dataset/ket_qua_absa.csv')

DS_FILE_GIAO_DIEN = {
    '/': ('web/index.html', 'text/html; charset=utf-8'),
    '/style.css': ('web/style.css', 'text/css; charset=utf-8'),
    '/script.js': ('web/script.js', 'application/javascript; charset=utf-8'),
}

CHU_THICH_BIEU_DO = {}
for ten_nhom, ds_bieu_do in hdc.DS_NHOM_BIEU_DO:
    for ten_file, chu_thich in ds_bieu_do:
        CHU_THICH_BIEU_DO[ten_file] = chu_thich

DS_ANH_BIEU_DO = list(CHU_THICH_BIEU_DO)
for ten_file in hdc.DS_BIEU_DO_DASHBOARD:
    DS_ANH_BIEU_DO.append(hdc.TIEN_TO_ANH_NHO + ten_file)


## 2. Vẽ biểu đồ

def so_sanh_mo_hinh() -> tuple:
    x_data = df['noi_dung_sach']
    y_data = df['label']
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                        test_size=hdc.TI_LE_TEST,
                                                        random_state=hdc.RANDOM_STATE,
                                                        stratify=y_data)
    pipe_nb = hdc.tao_pipeline_phan_loai(MultinomialNB())
    pipe_nb.fit(x_train, y_train)
    pipe_lr = hdc.tao_pipeline_phan_loai(LogisticRegression(max_iter=hdc.MAX_ITER, class_weight='balanced'))
    pipe_lr.fit(x_train, y_train)
    pipe_svc = hdc.tao_pipeline_phan_loai(LinearSVC(class_weight='balanced'))
    pipe_svc.fit(x_train, y_train)

    ds_ket_qua = []
    ds_ket_qua.append(hdc.danh_gia_mo_hinh('Naive Bayes', pipe_nb, x_train, y_train, x_test, y_test))
    ds_ket_qua.append(hdc.danh_gia_mo_hinh('Logistic Regression', pipe_lr, x_train, y_train, x_test, y_test))
    ds_ket_qua.append(hdc.danh_gia_mo_hinh('Linear SVC', pipe_svc, x_train, y_train, x_test, y_test))

    ds_nhan = [hdc.NHAN_TIEU_CUC, hdc.NHAN_TICH_CUC]
    ma_tran_lr = confusion_matrix(y_test, pipe_lr.predict(x_test), labels=ds_nhan)
    return pd.DataFrame(ds_ket_qua), ma_tran_lr


def ve_bieu_do():
    ax = hdc.tao_khung(7, 4.5)
    hdc.ve_phan_bo_cam_xuc(df, ax)
    hdc.luu_bieu_do('01_phan_bo_cam_xuc.png')

    ax = hdc.tao_khung(7, 4.5)
    hdc.ve_phan_bo_so_sao(df, ax)
    hdc.luu_bieu_do('02_phan_bo_so_sao.png')

    ax = hdc.tao_khung(7, 4.5)
    hdc.ve_cam_xuc_theo_sao(df, ax)
    hdc.luu_bieu_do('03_cam_xuc_theo_sao.png')

    ax = hdc.tao_khung(7, 4.5)
    hdc.ve_ti_le_che_theo_nganh(df, ax)
    hdc.luu_bieu_do('04_ti_le_che_theo_nganh.png')

    bang_so_sanh, ma_tran_lr = so_sanh_mo_hinh()
    ax = hdc.tao_khung(8, 4.5)
    hdc.ve_so_sanh_mo_hinh(bang_so_sanh, ax)
    hdc.luu_bieu_do('05_so_sanh_mo_hinh.png')

    ax = hdc.tao_khung(6, 4.5)
    hdc.ve_ma_tran_nham_lan(ma_tran_lr, ax)
    hdc.luu_bieu_do('06_ma_tran_nham_lan.png')

    bang_tu = hdc.lay_tu_quan_trong(mo_hinh, hdc.SO_TU_QUAN_TRONG)
    ax = hdc.tao_khung(8, 7)
    hdc.ve_tu_quan_trong(bang_tu, ax)
    hdc.luu_bieu_do('07_tu_quan_trong.png')

    ax = hdc.tao_khung(8, 5)
    hdc.ve_khen_che_khia_canh(df_absa, ax)
    hdc.luu_bieu_do('08_khen_che_khia_canh.png')

    ax = hdc.tao_khung(9, 4.5)
    hdc.ve_ti_le_che_khia_canh_nganh(df_absa, ax)
    hdc.luu_bieu_do('09_ti_le_che_khia_canh_nganh.png')

    ax = hdc.tao_khung(8, 5)
    hdc.ve_ti_le_che_theo_sao(df_absa, ax)
    hdc.luu_bieu_do('10_ti_le_che_theo_sao.png')

    ax = hdc.tao_khung(10, 5)
    hdc.ve_diem_khia_canh(df_absa, ax)
    hdc.luu_bieu_do('11_diem_khia_canh.png')

    bang_f1 = pd.read_csv('../dataset/ket_qua_f1_khia_canh.csv')
    ax = hdc.tao_khung(8, 5)
    hdc.ve_f1_khia_canh(bang_f1, ax)
    hdc.luu_bieu_do('12_f1_khia_canh.png')

    ax = hdc.tao_khung_nho()
    hdc.ve_khen_che_khia_canh(df_absa, ax)
    ax.set_title("Số bình luận khen và chê theo khía cạnh")
    ax.set_ylabel("")
    ax.legend(loc='upper right')
    hdc.thu_nho_so_tren_cot(ax)
    hdc.luu_bieu_do(hdc.TIEN_TO_ANH_NHO + '08_khen_che_khia_canh.png')

    ax = hdc.tao_khung_nho()
    hdc.ve_ti_le_che_khia_canh_nganh(df_absa, ax)
    ax.set_title("Tỉ lệ chê (%) theo khía cạnh và ngành hàng")
    ax.set_xlabel("")
    ax.set_ylabel("")
    for nhan in ax.get_xticklabels():
        nhan.set_horizontalalignment('right')
    hdc.luu_bieu_do(hdc.TIEN_TO_ANH_NHO + '09_ti_le_che_khia_canh_nganh.png')

    ax = hdc.tao_khung_nho()
    hdc.ve_ti_le_che_theo_sao(df_absa, ax)
    ax.set_title("Tỉ lệ chê của từng khía cạnh theo số sao")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    hdc.luu_bieu_do(hdc.TIEN_TO_ANH_NHO + '10_ti_le_che_theo_sao.png')

    ax = hdc.tao_khung_nho()
    hdc.ve_phan_bo_so_sao(df, ax)
    ax.margins(y=0.15)
    hdc.luu_bieu_do(hdc.TIEN_TO_ANH_NHO + '02_phan_bo_so_sao.png')

    ax = hdc.tao_khung_nho()
    hdc.ve_cam_xuc_theo_sao(df, ax)
    ax.set_ylim(0, 120)
    ax.legend(title="Cảm xúc", loc='center left', bbox_to_anchor=(1, 0.5))
    hdc.thu_nho_so_tren_cot(ax)
    hdc.luu_bieu_do(hdc.TIEN_TO_ANH_NHO + '03_cam_xuc_theo_sao.png')

    ax = hdc.tao_khung_nho()
    hdc.ve_ti_le_che_theo_nganh(df, ax)
    ax.set_ylabel("")
    hdc.luu_bieu_do(hdc.TIEN_TO_ANH_NHO + '04_ti_le_che_theo_nganh.png')

    ax = hdc.tao_khung_nho()
    hdc.ve_so_sanh_mo_hinh(bang_so_sanh, ax)
    ax.set_ylabel("")
    hdc.luu_bieu_do(hdc.TIEN_TO_ANH_NHO + '05_so_sanh_mo_hinh.png')

    bang_tu = hdc.lay_tu_quan_trong(mo_hinh, hdc.SO_TU_QUAN_TRONG_NHO)
    ax = hdc.tao_khung_nho()
    hdc.ve_tu_quan_trong(bang_tu, ax)
    ax.set_xlabel("Hệ số của mô hình (âm: tiêu cực, dương: tích cực)")
    ax.set_ylabel("")
    ax.tick_params(axis='y', labelsize=hdc.CO_SO_TREN_COT_NHO)
    hdc.luu_bieu_do(hdc.TIEN_TO_ANH_NHO + '07_tu_quan_trong.png')

    ax = hdc.tao_khung_nho()
    hdc.ve_f1_khia_canh(bang_f1, ax)
    ax.set_title("F1 nhận diện từng khía cạnh trên các câu gán tay")
    ax.set_ylabel("")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim(0, 1.15)
    hdc.thu_nho_so_tren_cot(ax)
    hdc.luu_bieu_do(hdc.TIEN_TO_ANH_NHO + '12_f1_khia_canh.png')

    print("Đã lưu", len(DS_ANH_BIEU_DO), "ảnh biểu đồ vào thư mục bieu_do/")


## 3. Kiểm tra đầu vào và dự đoán

def kiem_tra_binh_luan(binh_luan) -> str:
    if not isinstance(binh_luan, str):
        return "Bình luận phải là một đoạn chữ."
    if binh_luan.strip() == '':
        return "Bạn chưa nhập bình luận."
    if len(binh_luan) > hdc.DO_DAI_BINH_LUAN_TOI_DA:
        return "Bình luận dài quá " + str(hdc.DO_DAI_BINH_LUAN_TOI_DA) + " ký tự."
    return ''


def kiem_tra_so_sao(so_sao) -> str:
    if so_sao is None:
        return ''
    if isinstance(so_sao, bool) or not isinstance(so_sao, int):
        return "Số sao phải là số nguyên từ 1 đến 5."
    if so_sao < 1 or so_sao > 5:
        return "Số sao phải là số nguyên từ 1 đến 5."
    return ''


def xu_ly_du_doan(noi_dung_gui_len: bytes) -> tuple:
    try:
        du_lieu = json.loads(noi_dung_gui_len.decode('utf-8'))
    except ValueError:
        return 400, {'loi': "Dữ liệu gửi lên không đúng định dạng JSON."}
    if not isinstance(du_lieu, dict):
        return 400, {'loi': "Dữ liệu gửi lên không đúng định dạng JSON."}

    binh_luan = du_lieu.get('binh_luan')
    loi = kiem_tra_binh_luan(binh_luan)
    if loi != '':
        return 400, {'loi': loi}

    so_sao = du_lieu.get('so_sao')
    loi = kiem_tra_so_sao(so_sao)
    if loi != '':
        return 400, {'loi': loi}

    ket_qua = hdc.du_doan(binh_luan, mo_hinh)
    noi_dung_sach = ket_qua['noi_dung_da_lam_sach']
    if noi_dung_sach == '':
        return 400, {'loi': "Bình luận không có chữ nào để phân tích."}

    ket_qua['tu_anh_huong'] = hdc.lay_tu_anh_huong(mo_hinh, noi_dung_sach, hdc.SO_TU_ANH_HUONG)

    ket_qua['so_sao'] = so_sao
    if so_sao is None:
        ket_qua['doi_chieu_sao'] = ''
    else:
        ket_qua['doi_chieu_sao'] = hdc.doi_chieu_sao(so_sao, ket_qua['nhan'])
    ket_qua['khia_canh'] = lay_khia_canh(binh_luan)
    return 200, ket_qua


def lay_khia_canh(binh_luan: str) -> list:
    ket_qua_absa = hdc.phan_tich_absa(binh_luan, mo_hinh_cam_xuc_absa)
    ds_khia_canh = []
    for khia_canh in ket_qua_absa['aspects']:
        if khia_canh['sentiment'] == hdc.NHAN_TICH_CUC:
            cam_xuc = hdc.TEN_TICH_CUC
        else:
            cam_xuc = hdc.TEN_TIEU_CUC
        ds_khia_canh.append({'ten': hdc.TEN_KHIA_CANH_ABSA[khia_canh['aspect']],
                             'cam_xuc': cam_xuc,
                             'xac_suat_tich_cuc': khia_canh['xac_suat_tich_cuc']})
    return ds_khia_canh


def lay_du_lieu_dashboard() -> dict:
    ds_bieu_do = []
    for ten_file in hdc.DS_BIEU_DO_DASHBOARD:
        ds_bieu_do.append({'ten_file': ten_file,
                           'ten_file_nho': hdc.TIEN_TO_ANH_NHO + ten_file,
                           'chu_thich': CHU_THICH_BIEU_DO[ten_file]})
    return {'ds_bieu_do': ds_bieu_do}


## 4. Xử lý yêu cầu từ trình duyệt

class XuLyYeuCau(BaseHTTPRequestHandler):

    def gui_tra_loi(self, ma_trang_thai: int, kieu_noi_dung: str, noi_dung: bytes):
        self.send_response(ma_trang_thai)
        self.send_header('Content-Type', kieu_noi_dung)
        self.send_header('Content-Length', str(len(noi_dung)))
        self.end_headers()
        self.wfile.write(noi_dung)

    def gui_json(self, ma_trang_thai: int, du_lieu: dict):
        noi_dung = json.dumps(du_lieu, ensure_ascii=False).encode('utf-8')
        self.gui_tra_loi(ma_trang_thai, 'application/json; charset=utf-8', noi_dung)

    def do_GET(self):
        if self.path == '/api/dashboard':
            self.gui_json(200, lay_du_lieu_dashboard())
            return

        if self.path.startswith('/bieu_do/'):
            ten_file = self.path.replace('/bieu_do/', '', 1)
            if ten_file not in DS_ANH_BIEU_DO:
                self.gui_json(404, {'loi': "Không tìm thấy biểu đồ."})
                return
            with open('../bieu_do/' + ten_file, 'rb') as f:
                noi_dung = f.read()
            self.gui_tra_loi(200, 'image/png', noi_dung)
            return

        if self.path not in DS_FILE_GIAO_DIEN:
            self.gui_json(404, {'loi': "Không tìm thấy trang."})
            return
        ten_file, kieu_noi_dung = DS_FILE_GIAO_DIEN[self.path]
        with open(ten_file, 'rb') as f:
            noi_dung = f.read()
        self.gui_tra_loi(200, kieu_noi_dung, noi_dung)

    def do_POST(self):
        if self.path != '/api/du-doan':
            self.gui_json(404, {'loi': "Không tìm thấy trang."})
            return

        so_byte_toi_da = hdc.DO_DAI_BINH_LUAN_TOI_DA * 4 + 1000
        so_byte = int(self.headers.get('Content-Length', 0))
        if so_byte > so_byte_toi_da:
            self.gui_json(400, {'loi': "Dữ liệu gửi lên quá lớn."})
            return

        noi_dung_gui_len = self.rfile.read(so_byte)
        ma_trang_thai, ket_qua = xu_ly_du_doan(noi_dung_gui_len)
        self.gui_json(ma_trang_thai, ket_qua)


## 5. Chạy

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'web':
        print("Bỏ qua bước vẽ biểu đồ.")
    else:
        ve_bieu_do()

    may_chu = HTTPServer(('127.0.0.1', hdc.CONG_WEB), XuLyYeuCau)
    print("Web đang chạy tại: http://localhost:" + str(hdc.CONG_WEB))
    print("Bấm Ctrl + C để dừng.")
    try:
        may_chu.serve_forever()
    except KeyboardInterrupt:
        print("Đã dừng web.")
