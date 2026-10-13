import json
import os
import re
import sys
import unicodedata
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


## ==================== PHẦN A - HẰNG SỐ ====================

## 1. Chia dữ liệu và tính tái lập
RANDOM_STATE = 0
TI_LE_TEST = 0.3

## 2. Nhãn cảm xúc
NHAN_TICH_CUC = 'positive'
NHAN_TIEU_CUC = 'negative'
TEN_TICH_CUC = 'Tích cực'
TEN_TRUNG_LAP = 'Trung lập'
TEN_TIEU_CUC = 'Tiêu cực'

## 3. Ngưỡng xác suất để tạo nhãn Trung lập
NGUONG_DUOI = 0.20
NGUONG_TREN = 0.80

## 4. Tham số TF-IDF
NGRAM_RANGE = (1, 2)
MIN_DF = 2
SUBLINEAR_TF = True
MAX_ITER = 1000

## 5. Tiêu chí Shopee
DO_DAI_TEN_TIEU_CHI = 30
DAU_NGAN_TIEU_CHI = '  '

## 6. Từ điển teencode: từ viết tắt -> từ đầy đủ
TEENCODE = {
    'ko': 'không', 'k': 'không', 'kh': 'không', 'khong': 'không', 'hok': 'không',
    'hong': 'không', 'hông': 'không', 'khg': 'không', 'hk': 'không',
    'dc': 'được', 'đc': 'được', 'duoc': 'được',
    'sp': 'sản phẩm', 'sd': 'sử dụng', 'hsd': 'hạn sử dụng', 'srm': 'sữa rửa mặt',
    'bt': 'bình thường', 'bth': 'bình thường', 'bthg': 'bình thường',
    'vs': 'với', 'j': 'gì', 'r': 'rồi', 'z': 'vậy',
    'mk': 'mình', 'mik': 'mình', 'mn': 'mọi người', 'mng': 'mọi người', 'ng': 'người',
    'nt': 'nhắn tin', 'cx': 'cũng', 'trc': 'trước', 'lun': 'luôn', 'lm': 'làm',
    'ntn': 'như thế nào', 'chx': 'chưa', 'nhìu': 'nhiều', 'nhiu': 'nhiều',
    'nma': 'nhưng mà', 'ncl': 'nói chung là', 'nch': 'nói chung', 'siu': 'siêu',
    'oke': 'ok', 'oki': 'ok', 'okie': 'ok',
    'tks': 'cảm ơn', 'thanks': 'cảm ơn', 'thank': 'cảm ơn',
    'ship': 'giao hàng', 'sốp': 'shop',
    'tot': 'tốt', 'dep': 'đẹp', 'thom': 'thơm',
}

## 7. Ngành hàng
NGANH_KHONG_RO = 'Không rõ'

## 8. Web
DO_DAI_BINH_LUAN_TOI_DA = 2000
CONG_WEB = 8000
SO_TU_ANH_HUONG = 8

## 9. Biểu đồ
DPI_BIEU_DO = 150
MAU_TICH_CUC = '#2e9e5b'
MAU_TIEU_CUC = '#d64545'
MAU_CHUNG = '#1f3a5f'

MAU_CAM_XUC = {TEN_TICH_CUC: MAU_TICH_CUC, TEN_TIEU_CUC: MAU_TIEU_CUC}

SO_TU_QUAN_TRONG = 10

## 10. Emoji mang cảm xúc (dùng cho ABSA)
DS_EMOJI_TICH_CUC = ['🥰', '❤', '👍', '😍', '🤣', '💯', '🫶', '😘', '☺', '🌟', '😚']
DS_EMOJI_TIEU_CUC = ['🥲', '😡', '😭', '🙂', '🥹', '⭐', '😢', '😀', '😌', '🙃',
                     '👎', '😑', '🙄', '😓', '😃', '😞', '😥', '😤', '🙏', '🥺']
TU_EMOJI_TICH_CUC = 'emoji_tich_cuc'
TU_EMOJI_TIEU_CUC = 'emoji_tieu_cuc'

## 11. Sáu khía cạnh của ABSA
DS_KHIA_CANH = ['PRODUCT_QUALITY', 'DELIVERY', 'PACKAGING', 'PRICE', 'SELLER_SERVICE', 'PRODUCT_FEATURE']

TEN_KHIA_CANH_ABSA = {
    'PRODUCT_QUALITY': 'Chất lượng sản phẩm',
    'DELIVERY': 'Giao hàng',
    'PACKAGING': 'Đóng gói',
    'PRICE': 'Giá',
    'SELLER_SERVICE': 'Dịch vụ của shop',
    'PRODUCT_FEATURE': 'Thuộc tính sản phẩm',
}

TU_KHOA_ABSA = {
    'PRODUCT_QUALITY': ['chất lượng', 'hàng giả', 'fake', 'chính hãng', 'hỏng', 'hư', 'lỗi', 'bền',
                        'date', 'hạn sử dụng', 'hết hạn', 'hiệu quả', 'tác dụng', 'công dụng',
                        'sản phẩm tốt', 'sản phẩm ok', 'sản phẩm đẹp', 'hàng tốt', 'hàng ok', 'hàng đẹp',
                        'dùng tốt', 'dùng ok', 'dùng thích', 'xài tốt', 'xài ok',
                        'nên mua', 'đáng mua', 'mua lại', 'đúng mô tả', 'như mô tả', 'như quảng cáo',
                        'hài lòng', 'ưng', 'ưng ý', 'tuyệt vời', 'xuất sắc', 'như hình', 'giống hình', 'như ảnh',
                        'giống mô tả', 'hàng thật', 'hàng dỏm', 'nứt', 'sản phẩm ổn', 'dùng ổn', 'xài ổn',
                        'không dùng được'],
    'DELIVERY': ['giao hàng', 'giao nhanh', 'giao chậm', 'giao lâu', 'giao trễ', 'giao siêu nhanh', 'giao rất nhanh',
                 'giao khá nhanh', 'giao cũng nhanh', 'giao hơi lâu', 'giao quá lâu', 'ship', 'shipper', 'tài xế',
                 'vận chuyển', 'đơn vị vận chuyển', 'hỏa tốc', 'hoả tốc', 'hàng về', 'gửi hàng', 'nhận hàng nhanh'],
    'PACKAGING': ['đóng gói', 'gói hàng', 'đóng hàng', 'gói kỹ', 'gói kĩ', 'gói cẩn thận', 'hộp', 'móp', 'méo',
                  'bọc', 'chống sốc', 'bao bì', 'thùng hàng', 'thùng carton', 'thùng giấy',
                  'bể', 'vỡ', 'rách', 'chảy', 'seal', 'niêm phong'],
    'PRICE': ['giá', 'rẻ', 'đắt', 'mắc', 'giá tiền', 'đáng tiền', 'phí tiền', 'tiếc tiền', 'túi tiền',
              'sale', 'giảm giá', 'mã giảm', 'voucher'],
    'SELLER_SERVICE': ['tư vấn', 'nhắn tin', 'trả lời', 'phản hồi', 'chăm sóc khách hàng', 'cskh',
                       'nhân viên', 'hỗ trợ', 'rep', 'thái độ', 'uy tín', 'nhiệt tình',
                       'đổi trả', 'trả hàng', 'hoàn tiền', 'hoàn hàng',
                       'quà', 'tặng', 'khuyến mãi', 'giao thiếu', 'giao sai', 'thiếu hàng', 'sai hàng',
                       'dịch vụ', 'phục vụ', 'chăm sóc', 'chu đáo', 'thờ ơ', 'làm ăn', 'làm việc', 'trách nhiệm',
                       'giải quyết', 'lừa', 'lừa đảo', 'chặn', 'hủy đơn', 'huỷ đơn', 'deal', 'giao nhầm', 'giao đủ',
                       'đổi size', 'bảo hành'],
    'PRODUCT_FEATURE': ['mùi', 'thơm', 'hương', 'màu', 'hương vị', 'mùi vị', 'ngon', 'size', 'kích thước',
                        'kích cỡ', 'form', 'chất liệu', 'vải', 'chất kem', 'chất son', 'thiết kế', 'mẫu mã',
                        'kiểu dáng', 'thấm', 'mềm', 'mịn', 'mượt', 'bết', 'kích ứng', 'dưỡng ẩm', 'cấp ẩm',
                        'làm sạch', 'pin', 'âm thanh'],
}

DS_CUM_TU_GAY_NHAM = ['đánh giá']

TIEU_CHI_KHIA_CANH = {
    'PACKAGING': ['Bao bì', 'Bao bì/Mẫu mã', 'Hộp'],
    'PRODUCT_QUALITY': ['Chất lượng sản phẩm', 'Chất lượng', 'Đúng với mô tả', 'Hiệu quả', 'Độ bền'],
}
DS_TIEU_CHI_BO_QUA = ['Kinh nghiệm sử dụng', 'Mẹo', 'Độ tuổi sử dụng', 'Đối tượng sử dụng',
                      'Độ tuổi', 'Lưu ý quan trọng']
KHIA_CANH_TIEU_CHI_CON_LAI = 'PRODUCT_FEATURE'

## 12. Cắt bình luận thành vế và chấm cảm xúc từng vế (ABSA)
DS_TU_CAT_VE = ['tuy nhiên', 'nhưng mà', 'nhưng', 'tuy', 'mỗi tội', 'có điều', 'ngoại trừ', 'nma', 'nhg']
MAU_DAU_CAT_VE = r'[.,;!?\n]+'
NGUONG_KHEN_CHE = 0.5

## 13. Trang Dashboard trên web
DS_NHOM_BIEU_DO = [
    ('Tổng quan', [
        ('01_phan_bo_cam_xuc.png', "Bình luận tiêu cực nhiều hơn tích cực, dữ liệu lệch về phía chê."),
        ('02_phan_bo_so_sao.png', "Khách chấm 5 sao và 1 sao nhiều nhất, 4 sao rất ít."),
        ('03_cam_xuc_theo_sao.png', "1, 2 và 3 sao hầu hết là chê; 5 sao hầu hết là khen."),
        ('04_ti_le_che_theo_nganh.png', "Tỉ lệ chê của từng ngành hàng so với mức chung."),
    ]),
    ('Mô hình', [
        ('05_so_sanh_mo_hinh.png', "Điểm của 3 mô hình phân loại trên tập test."),
        ('06_ma_tran_nham_lan.png', "Số câu Logistic Regression đoán đúng và đoán sai."),
        ('07_tu_quan_trong.png', "Các từ kéo mô hình về phía tiêu cực và tích cực mạnh nhất."),
    ]),
    ('Khía cạnh (ABSA)', [
        ('08_khen_che_khia_canh.png', "Dịch vụ của shop và đóng gói bị chê nhiều hơn khen; 4 khía cạnh còn lại được khen nhiều hơn."),
        ('09_ti_le_che_khia_canh_nganh.png', "Ô càng đỏ thì khía cạnh đó càng bị chê nhiều ở ngành hàng đó."),
        ('10_ti_le_che_theo_sao.png', "Dịch vụ của shop và đóng gói vẫn bị chê quá nửa ở mức 4 sao."),
        ('11_diem_khia_canh.png', "Điểm thang 10 = tỉ lệ bình luận khen; đường đứt nét là mức 5."),
        ('12_f1_khia_canh.png', "Độ đúng khi nhận diện từng khía cạnh, chấm trên các câu gán tay."),
    ]),
]

DS_BIEU_DO_DASHBOARD = [
    '08_khen_che_khia_canh.png',
    '09_ti_le_che_khia_canh_nganh.png',
    '10_ti_le_che_theo_sao.png',
    '02_phan_bo_so_sao.png',
    '03_cam_xuc_theo_sao.png',
    '04_ti_le_che_theo_nganh.png',
    '05_so_sanh_mo_hinh.png',
    '07_tu_quan_trong.png',
    '12_f1_khia_canh.png',
]
TIEN_TO_ANH_NHO = 'nho_'
RONG_ANH_NHO = 5.2
CAO_ANH_NHO = 2.2
CO_CHU_ANH_NHO = 0.8
SO_TU_QUAN_TRONG_NHO = 5
CO_SO_TREN_COT_NHO = 7


## ==================== PHẦN B - XỬ LÝ VĂN BẢN ====================

## 1. Tách tiêu chí Shopee

def la_ten_tieu_chi(ten: str) -> bool:
    if len(ten) == 0 or len(ten) > DO_DAI_TEN_TIEU_CHI:
        return False
    if not ten[0].isupper():
        return False
    if ten[-1] == ' ':
        return False
    for ky_tu in ten:
        if not (ky_tu.isalpha() or ky_tu == ' ' or ky_tu == '/'):
            return False
    return True


def tach_tieu_chi(review: str) -> tuple:
    vi_tri = review.find(':')
    if vi_tri == -1:
        return '', review
    if not la_ten_tieu_chi(review[:vi_tri]):
        return '', review

    ds_phan = review.split(DAU_NGAN_TIEU_CHI, 1)
    tieu_chi = ds_phan[0].strip()
    if len(ds_phan) == 2:
        noi_dung = ds_phan[1].strip()
    else:
        noi_dung = ''
    return tieu_chi, noi_dung


def tach_ten_va_gia_tri(tieu_chi: str) -> tuple:
    ds_ten = []
    ds_gia_tri = []
    if tieu_chi == '':
        return ds_ten, ds_gia_tri

    ds_doan = tieu_chi.split(':')
    ds_ten.append(ds_doan[0])

    for doan in ds_doan[1:-1]:
        ds_tu = doan.split(' ')
        vi_tri_ten = -1
        for i in range(len(ds_tu)):
            if ds_tu[i] != '' and ds_tu[i][0].isupper():
                vi_tri_ten = i
        if vi_tri_ten <= 0:
            ds_gia_tri.append(doan)
            ds_ten.append('')
        else:
            ds_gia_tri.append(' '.join(ds_tu[:vi_tri_ten]))
            ds_ten.append(' '.join(ds_tu[vi_tri_ten:]))

    if len(ds_doan) > 1:
        ds_gia_tri.append(ds_doan[-1])

    ds_ten = [ten for ten in ds_ten if ten != '']
    return ds_ten, ds_gia_tri


def lay_gia_tri_tieu_chi(tieu_chi: str) -> str:
    ds_ten, ds_gia_tri = tach_ten_va_gia_tri(tieu_chi)
    return ' '.join(ds_gia_tri)


## 2. Làm sạch văn bản

def chuan_hoa_unicode(chuoi: str) -> str:
    return unicodedata.normalize('NFC', chuoi)


def bo_link(chuoi: str) -> str:
    return re.sub(r'http\S+|www\.\S+', ' ', chuoi)


def bo_ky_tu_dac_biet(chuoi: str) -> str:
    chuoi = re.sub(r'[^\w\s]', ' ', chuoi)
    return chuoi.replace('_', ' ')


def chuan_hoa_teencode(chuoi: str) -> str:
    ds_tu_moi = []
    for tu in chuoi.split():
        ds_tu_moi.append(TEENCODE.get(tu, tu))
    return ' '.join(ds_tu_moi)


def bo_khoang_trang_thua(chuoi: str) -> str:
    return ' '.join(chuoi.split())


def lam_sach(chuoi: str) -> str:
    chuoi = chuan_hoa_unicode(chuoi)
    chuoi = chuoi.lower()
    chuoi = bo_link(chuoi)
    chuoi = bo_ky_tu_dac_biet(chuoi)
    chuoi = chuan_hoa_teencode(chuoi)
    chuoi = bo_khoang_trang_thua(chuoi)
    return chuoi


def lam_sach_binh_luan(binh_luan: str) -> str:
    tieu_chi, noi_dung = tach_tieu_chi(binh_luan)
    gia_tri = lay_gia_tri_tieu_chi(tieu_chi)
    return lam_sach(gia_tri + ' ' + noi_dung)


## 3. Tìm từ khóa

def co_tu_khoa(chuoi: str, ds_tu_khoa: list) -> int:
    chuoi_co_dem = ' ' + chuoi + ' '
    for tu_khoa in ds_tu_khoa:
        if ' ' + tu_khoa + ' ' in chuoi_co_dem:
            return 1
    return 0


## 4. Dự đoán

def gan_nhan(p: float, nguong_duoi: float = NGUONG_DUOI,
             nguong_tren: float = NGUONG_TREN) -> str:
    if p > nguong_tren:
        return TEN_TICH_CUC
    if p < nguong_duoi:
        return TEN_TIEU_CUC
    return TEN_TRUNG_LAP


def doi_chieu_sao(so_sao: int, nhan: str) -> str:
    if so_sao >= 4:
        nhan_theo_sao = TEN_TICH_CUC
    elif so_sao <= 2:
        nhan_theo_sao = TEN_TIEU_CUC
    else:
        nhan_theo_sao = TEN_TRUNG_LAP

    if nhan_theo_sao == nhan:
        return 'Khớp'
    if nhan_theo_sao == TEN_TRUNG_LAP or nhan == TEN_TRUNG_LAP:
        return 'Lệch nhẹ'
    return 'Lệch sao'


def tinh_xac_suat_tich_cuc(mo_hinh, ds_van_ban) -> list:
    ds_lop = list(mo_hinh.classes_)
    vi_tri_tich_cuc = ds_lop.index(NHAN_TICH_CUC)
    bang_xac_suat = mo_hinh.predict_proba(ds_van_ban)
    return bang_xac_suat[:, vi_tri_tich_cuc]


def du_doan(binh_luan: str, mo_hinh) -> dict:
    noi_dung_sach = lam_sach_binh_luan(binh_luan)
    ds_xac_suat = tinh_xac_suat_tich_cuc(mo_hinh, [noi_dung_sach])
    p = float(ds_xac_suat[0])
    ket_qua = {
        'nhan': gan_nhan(p),
        'xac_suat_tich_cuc': round(p, 4),
        'noi_dung_da_lam_sach': noi_dung_sach,
    }
    return ket_qua


## 5. Làm sạch cho ABSA

def doi_emoji(chuoi: str) -> str:
    for emoji in DS_EMOJI_TICH_CUC:
        chuoi = chuoi.replace(emoji, ' ' + TU_EMOJI_TICH_CUC + ' ')
    for emoji in DS_EMOJI_TIEU_CUC:
        chuoi = chuoi.replace(emoji, ' ' + TU_EMOJI_TIEU_CUC + ' ')
    return chuoi


def rut_ky_tu_lap(chuoi: str) -> str:
    return re.sub(r'([a-zà-ỹ])\1{2,}', r'\1', chuoi)


def lam_sach_absa(chuoi: str) -> str:
    chuoi = chuan_hoa_unicode(chuoi)
    chuoi = chuoi.lower()
    chuoi = bo_link(chuoi)
    chuoi = doi_emoji(chuoi)
    chuoi = re.sub(r'[^\w\s]', ' ', chuoi)
    chuoi = chuan_hoa_teencode(chuoi)
    chuoi = rut_ky_tu_lap(chuoi)
    chuoi = bo_khoang_trang_thua(chuoi)
    return chuoi


def lam_sach_binh_luan_absa(binh_luan: str) -> str:
    tieu_chi, noi_dung = tach_tieu_chi(binh_luan)
    gia_tri = lay_gia_tri_tieu_chi(tieu_chi)
    return lam_sach_absa(gia_tri + ' ' + noi_dung)


## ==================== PHẦN C - MÔ HÌNH ====================

## 1. Xây dựng mô hình

def tao_pipeline_phan_loai(bo_phan_loai):
    buoc_tfidf = TfidfVectorizer(ngram_range=NGRAM_RANGE,
                                 min_df=MIN_DF,
                                 sublinear_tf=SUBLINEAR_TF)
    return Pipeline([('tfidf', buoc_tfidf), ('mo_hinh', bo_phan_loai)])


## 2. Đánh giá mô hình

def danh_gia_mo_hinh(ten: str, mo_hinh, x_train, y_train, x_test, y_test) -> dict:
    y_du_doan = mo_hinh.predict(x_test)
    ket_qua = {
        'Mô hình': ten,
        'Accuracy train': round(mo_hinh.score(x_train, y_train), 4),
        'Accuracy test': round(accuracy_score(y_test, y_du_doan), 4),
        'F1 negative': round(f1_score(y_test, y_du_doan, pos_label=NHAN_TIEU_CUC), 4),
        'F1 positive': round(f1_score(y_test, y_du_doan, pos_label=NHAN_TICH_CUC), 4),
        'Macro-F1': round(f1_score(y_test, y_du_doan, average='macro'), 4),
    }
    return ket_qua


def lay_tu_quan_trong(mo_hinh, so_tu: int):
    ds_tu = mo_hinh.named_steps['tfidf'].get_feature_names_out()
    ds_he_so = mo_hinh.named_steps['mo_hinh'].coef_[0]
    bang = pd.DataFrame({'tu': ds_tu, 'he_so': ds_he_so})
    bang = bang.sort_values('he_so')

    bang_tieu_cuc = bang.head(so_tu).reset_index(drop=True)
    bang_tich_cuc = bang.tail(so_tu).iloc[::-1].reset_index(drop=True)
    ket_qua = pd.DataFrame({
        'Từ kéo về tiêu cực': bang_tieu_cuc['tu'],
        'Hệ số âm': bang_tieu_cuc['he_so'].round(2),
        'Từ kéo về tích cực': bang_tich_cuc['tu'],
        'Hệ số dương': bang_tich_cuc['he_so'].round(2),
    })
    return ket_qua


## 3. Giải thích dự đoán

def lay_tu_anh_huong(mo_hinh, van_ban: str, so_tu: int) -> list:
    buoc_tfidf = mo_hinh.named_steps['tfidf']
    ds_tu = buoc_tfidf.get_feature_names_out()
    ds_he_so = mo_hinh.named_steps['mo_hinh'].coef_[0]

    vector = buoc_tfidf.transform([van_ban])
    bang = pd.DataFrame({'tu': ds_tu[vector.indices],
                         'dong_gop': vector.data * ds_he_so[vector.indices]})
    bang['do_lon'] = bang['dong_gop'].abs()
    bang = bang.sort_values('do_lon', ascending=False).head(so_tu)

    ket_qua = []
    for i in range(len(bang)):
        ket_qua.append({'tu': bang['tu'].iloc[i], 'dong_gop': round(float(bang['dong_gop'].iloc[i]), 4)})
    return ket_qua


## ==================== PHẦN D - KHÍA CẠNH ====================

## 1. Nhận diện khía cạnh

def tim_khia_canh_tu_khoa(van_ban: str) -> list:
    for cum_tu in DS_CUM_TU_GAY_NHAM:
        van_ban = van_ban.replace(cum_tu, ' ')

    ds_khia_canh = []
    for ma in DS_KHIA_CANH:
        if co_tu_khoa(van_ban, TU_KHOA_ABSA[ma]) == 1:
            ds_khia_canh.append(ma)
    return ds_khia_canh


def tim_khia_canh_tieu_chi(tieu_chi: str) -> list:
    ds_ten, ds_gia_tri = tach_ten_va_gia_tri(tieu_chi)
    ds_khia_canh = []
    for ten in ds_ten:
        if ten in DS_TIEU_CHI_BO_QUA:
            continue
        ma = KHIA_CANH_TIEU_CHI_CON_LAI
        for ma_khia_canh, ds_ten_tieu_chi in TIEU_CHI_KHIA_CANH.items():
            if ten in ds_ten_tieu_chi:
                ma = ma_khia_canh
        if ma not in ds_khia_canh:
            ds_khia_canh.append(ma)
    return ds_khia_canh


def tim_khia_canh(binh_luan: str) -> list:
    tieu_chi, noi_dung = tach_tieu_chi(binh_luan)
    ds_theo_tu_khoa = tim_khia_canh_tu_khoa(lam_sach_binh_luan_absa(binh_luan))
    ds_theo_tieu_chi = tim_khia_canh_tieu_chi(tieu_chi)
    return [ma for ma in DS_KHIA_CANH if ma in ds_theo_tu_khoa or ma in ds_theo_tieu_chi]


## 2. Cảm xúc theo khía cạnh

def cat_ve_cau(noi_dung: str) -> list:
    noi_dung = chuan_hoa_unicode(noi_dung).lower()
    for tu in DS_TU_CAT_VE:
        noi_dung = re.sub(r'\b' + tu + r'\b', '.', noi_dung)

    ds_ve = []
    for ve in re.split(MAU_DAU_CAT_VE, noi_dung):
        if ve.strip() != '':
            ds_ve.append(ve.strip())
    return ds_ve


def tach_ve_binh_luan(binh_luan: str) -> list:
    tieu_chi, noi_dung = tach_tieu_chi(binh_luan)
    ds_ve = []

    ds_ten, ds_gia_tri = tach_ten_va_gia_tri(tieu_chi)
    if len(ds_ten) == len(ds_gia_tri):
        for i in range(len(ds_ten)):
            van_ban = lam_sach_absa(ds_gia_tri[i])
            ds_khia_canh = tim_khia_canh_tieu_chi(ds_ten[i] + ':')
            for ma in tim_khia_canh_tu_khoa(van_ban):
                if ma not in ds_khia_canh:
                    ds_khia_canh.append(ma)
            if van_ban != '' and len(ds_khia_canh) > 0:
                ds_ve.append({'ve': van_ban, 'ds_khia_canh': ds_khia_canh})

    for ve in cat_ve_cau(noi_dung):
        van_ban = lam_sach_absa(ve)
        ds_khia_canh = tim_khia_canh_tu_khoa(van_ban)
        if len(ds_khia_canh) > 0:
            ds_ve.append({'ve': van_ban, 'ds_khia_canh': ds_khia_canh})
    return ds_ve


def cham_cam_xuc_khia_canh(binh_luan: str, mo_hinh_cam_xuc) -> dict:
    ds_ve = tach_ve_binh_luan(binh_luan)
    if len(ds_ve) == 0:
        return {}

    ds_van_ban = [ve['ve'] for ve in ds_ve]
    ds_xac_suat = tinh_xac_suat_tich_cuc(mo_hinh_cam_xuc, ds_van_ban)

    xac_suat_theo_khia_canh = {}
    for i in range(len(ds_ve)):
        for ma in ds_ve[i]['ds_khia_canh']:
            if ma not in xac_suat_theo_khia_canh:
                xac_suat_theo_khia_canh[ma] = []
            xac_suat_theo_khia_canh[ma].append(float(ds_xac_suat[i]))

    ket_qua = {}
    for ma in DS_KHIA_CANH:
        if ma in xac_suat_theo_khia_canh:
            ds_p = xac_suat_theo_khia_canh[ma]
            ket_qua[ma] = round(sum(ds_p) / len(ds_p), 4)
    return ket_qua


## 3. Ghép ABSA

def phan_tich_absa(binh_luan: str, mo_hinh_cam_xuc) -> dict:
    ds_khia_canh = tim_khia_canh(binh_luan)
    xac_suat_theo_ve = cham_cam_xuc_khia_canh(binh_luan, mo_hinh_cam_xuc)
    ds_p_ca_cau = tinh_xac_suat_tich_cuc(mo_hinh_cam_xuc, [lam_sach_binh_luan_absa(binh_luan)])
    p_ca_cau = round(float(ds_p_ca_cau[0]), 4)

    ds_ket_qua = []
    for ma in DS_KHIA_CANH:
        if ma in xac_suat_theo_ve:
            p = xac_suat_theo_ve[ma]
        elif ma in ds_khia_canh:
            p = p_ca_cau
        else:
            continue
        if p >= NGUONG_KHEN_CHE:
            cam_xuc = NHAN_TICH_CUC
        else:
            cam_xuc = NHAN_TIEU_CUC
        ds_ket_qua.append({'aspect': ma, 'sentiment': cam_xuc, 'xac_suat_tich_cuc': p})
    return {'review': binh_luan, 'aspects': ds_ket_qua}


## ==================== PHẦN E - BIỂU ĐỒ ====================

## 1. Hàm phụ

def tao_khung(rong: float, cao: float):
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(rong, cao))
    return plt.gca()


def tao_khung_nho():
    sns.set_theme(style='whitegrid', font_scale=CO_CHU_ANH_NHO)
    plt.figure(figsize=(RONG_ANH_NHO, CAO_ANH_NHO))
    return plt.gca()


def thu_nho_so_tren_cot(ax):
    for chu in ax.texts:
        chu.set_fontsize(CO_SO_TREN_COT_NHO)


def luu_bieu_do(ten_file: str):
    os.makedirs('../bieu_do', exist_ok=True)
    plt.tight_layout()
    plt.savefig('../bieu_do/' + ten_file, dpi=DPI_BIEU_DO)
    plt.close()


def ghi_so_len_cot(ax, dinh_dang: str = '%d'):
    for nhom_cot in ax.containers:
        ax.bar_label(nhom_cot, fmt=dinh_dang, padding=2)


def ghi_so_giua_cot(ax, dinh_dang: str = '%d'):
    for nhom_cot in ax.containers:
        ax.bar_label(nhom_cot, fmt=dinh_dang, label_type='center', color='white', fontweight='bold')


def doi_ten_cam_xuc(ds_nhan):
    return ds_nhan.replace({NHAN_TICH_CUC: TEN_TICH_CUC,
                            NHAN_TIEU_CUC: TEN_TIEU_CUC})


def tinh_ti_le_tieu_cuc(df) -> float:
    return (df['label'] == NHAN_TIEU_CUC).mean() * 100


## 2. Nhóm Tổng quan

def ve_phan_bo_cam_xuc(df, ax):
    bang = pd.DataFrame({'cam_xuc': doi_ten_cam_xuc(df['label'])})
    sns.countplot(data=bang, x='cam_xuc', hue='cam_xuc', palette=MAU_CAM_XUC,
                  order=[TEN_TICH_CUC, TEN_TIEU_CUC], legend=False, ax=ax)
    ghi_so_len_cot(ax)
    ax.set_title("Phân bố cảm xúc")
    ax.set_xlabel("Nhãn cảm xúc")
    ax.set_ylabel("Số bình luận")


def ve_phan_bo_so_sao(df, ax):
    sns.countplot(data=df, x='rating', color=MAU_CHUNG, ax=ax)
    ghi_so_len_cot(ax)
    ax.set_title("Phân bố số sao")
    ax.set_xlabel("Số sao")
    ax.set_ylabel("Số bình luận")


def ve_cam_xuc_theo_sao(df, ax):
    bang = pd.crosstab(df['rating'], doi_ten_cam_xuc(df['label']), normalize='index') * 100
    bang = bang.reset_index()
    bang = bang.melt(id_vars='rating', var_name='label', value_name='ti_le')

    sns.barplot(data=bang, x='rating', y='ti_le', hue='label', palette=MAU_CAM_XUC,
                hue_order=[TEN_TICH_CUC, TEN_TIEU_CUC], ax=ax)
    ghi_so_len_cot(ax, '%.1f%%')
    ax.set_title("Cảm xúc theo từng mức sao")
    ax.set_xlabel("Số sao")
    ax.set_ylabel("Tỉ lệ bình luận (%)")
    ax.set_ylim(0, 140)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(title="Cảm xúc", loc='upper center', ncol=2)


def ve_ti_le_che_theo_nganh(df, ax):
    ds_nganh = []
    ds_ti_le = []
    for nganh in df['nganh_hang'].unique():
        df_nganh = df[df['nganh_hang'] == nganh]
        ds_nganh.append(nganh)
        ds_ti_le.append(tinh_ti_le_tieu_cuc(df_nganh))
    bang = pd.DataFrame({'nganh_hang': ds_nganh, 'ti_le': ds_ti_le})
    bang = bang.sort_values('ti_le', ascending=False)

    sns.barplot(data=bang, x='ti_le', y='nganh_hang', color=MAU_TIEU_CUC, ax=ax)
    ghi_so_giua_cot(ax, '%.1f%%')
    ax.axvline(tinh_ti_le_tieu_cuc(df), color='black', linestyle='--', label="Chung cả bộ dữ liệu")
    ax.set_title("Tỉ lệ bình luận tiêu cực theo ngành hàng")
    ax.set_xlabel("Tỉ lệ tiêu cực (%)")
    ax.set_ylabel("Ngành hàng")
    ax.set_xlim(0, 100)
    ax.legend(loc='lower right')


## 3. Nhóm Mô hình


def ve_tu_quan_trong(bang_tu, ax):
    bang_tich_cuc = pd.DataFrame({'tu': bang_tu['Từ kéo về tích cực'],
                                  'he_so': bang_tu['Hệ số dương'],
                                  'phia': TEN_TICH_CUC})
    bang_tieu_cuc = pd.DataFrame({'tu': bang_tu['Từ kéo về tiêu cực'],
                                  'he_so': bang_tu['Hệ số âm'],
                                  'phia': TEN_TIEU_CUC})
    bang = pd.concat([bang_tich_cuc, bang_tieu_cuc])

    sns.barplot(data=bang, x='he_so', y='tu', hue='phia', palette=MAU_CAM_XUC,
                dodge=False, ax=ax)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_title("Các từ ảnh hưởng mạnh nhất tới dự đoán")
    ax.set_xlabel("Hệ số của mô hình (âm: kéo về tiêu cực, dương: kéo về tích cực)")
    ax.set_ylabel("Từ hoặc cụm 2 từ")
    ax.legend(title="Kéo về phía")


def ve_so_sanh_mo_hinh(bang_so_sanh, ax):
    bang = bang_so_sanh[['Mô hình', 'Accuracy test', 'Macro-F1']]
    bang = bang.melt(id_vars='Mô hình', var_name='thang_do', value_name='diem')

    sns.barplot(data=bang, x='diem', y='Mô hình', hue='thang_do', ax=ax)
    ghi_so_len_cot(ax, '%.4f')
    ax.set_title("So sánh 3 mô hình trên tập test")
    ax.set_xlabel("Điểm (càng gần 1 càng tốt)")
    ax.set_ylabel("Mô hình")
    ax.set_xlim(0, 1.12)
    ax.legend(title="Thang đo", loc='center left', bbox_to_anchor=(1, 0.5))


def ve_ma_tran_nham_lan(ma_tran, ax):
    ds_ten = [TEN_TIEU_CUC, TEN_TICH_CUC]
    sns.heatmap(ma_tran, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=ds_ten, yticklabels=ds_ten, annot_kws={'size': 16}, ax=ax)
    ax.set_title("Ma trận nhầm lẫn của Logistic Regression")
    ax.set_xlabel("Mô hình đoán")
    ax.set_ylabel("Nhãn thật")


## 4. Nhóm Khía cạnh (ABSA)

def ve_khen_che_khia_canh(df_absa, ax):
    ds_dong = []
    for ma in DS_KHIA_CANH:
        ten = TEN_KHIA_CANH_ABSA[ma]
        so_khen = (df_absa['cx_' + ma] == NHAN_TICH_CUC).sum()
        so_che = (df_absa['cx_' + ma] == NHAN_TIEU_CUC).sum()
        ds_dong.append({'khia_canh': ten, 'cam_xuc': TEN_TICH_CUC, 'so_cau': so_khen})
        ds_dong.append({'khia_canh': ten, 'cam_xuc': TEN_TIEU_CUC, 'so_cau': so_che})
    bang = pd.DataFrame(ds_dong)

    sns.barplot(data=bang, x='so_cau', y='khia_canh', hue='cam_xuc', palette=MAU_CAM_XUC, ax=ax)
    ghi_so_len_cot(ax)
    ax.set_title("Số bình luận khen và chê theo từng khía cạnh (ABSA)")
    ax.set_xlabel("Số bình luận")
    ax.set_ylabel("Khía cạnh")
    ax.set_xlim(0, bang['so_cau'].max() * 1.45)
    ax.legend(title="Cảm xúc", loc='center right')


def ve_ti_le_che_khia_canh_nganh(df_absa, ax):
    df_ro_nganh = df_absa[df_absa['nganh_hang'] != NGANH_KHONG_RO]
    bang = pd.DataFrame()
    for ma in DS_KHIA_CANH:
        df_co_nhac = df_ro_nganh[df_ro_nganh['cx_' + ma].notna()].copy()
        df_co_nhac['la_che'] = df_co_nhac['cx_' + ma] == NHAN_TIEU_CUC
        bang[TEN_KHIA_CANH_ABSA[ma]] = df_co_nhac.groupby('nganh_hang')['la_che'].mean() * 100

    sns.heatmap(bang, annot=True, fmt=".0f", cmap="Reds", vmin=0, vmax=100,
                cbar_kws={'label': "Tỉ lệ chê (%)"}, ax=ax)
    ax.set_title("Tỉ lệ chê (%) của từng khía cạnh theo ngành hàng (ABSA)")
    ax.set_xlabel("Khía cạnh")
    ax.set_ylabel("Ngành hàng")
    ax.tick_params(axis='x', rotation=20)
    ax.tick_params(axis='y', rotation=0)


def ve_f1_khia_canh(bang_f1, ax):
    sns.barplot(data=bang_f1, x='f1', y='khia_canh', hue='cach', ax=ax)
    ghi_so_len_cot(ax, '%.2f')
    ax.set_title("F1 nhận diện từng khía cạnh trên các câu gán tay (ABSA)")
    ax.set_xlabel("F1 (càng gần 1 càng tốt)")
    ax.set_ylabel("Khía cạnh")
    ax.set_xlim(0, 1.45)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.legend(title="Cách nhận diện", loc='lower right')


def ve_ti_le_che_theo_sao(df_absa, ax):
    ds_dong = []
    for ma in DS_KHIA_CANH:
        df_co_nhac = df_absa[df_absa['cx_' + ma].notna()].copy()
        df_co_nhac['la_che'] = (df_co_nhac['cx_' + ma] == NHAN_TIEU_CUC) * 100
        ti_le_che = df_co_nhac.groupby('rating')['la_che'].mean()
        for so_sao in ti_le_che.index:
            ds_dong.append({'khia_canh': TEN_KHIA_CANH_ABSA[ma], 'rating': so_sao, 'ti_le_che': ti_le_che[so_sao]})
    bang = pd.DataFrame(ds_dong)

    sns.lineplot(data=bang, x='rating', y='ti_le_che', hue='khia_canh', marker='o', ax=ax)
    ax.set_title("Tỉ lệ chê của từng khía cạnh theo số sao (ABSA)")
    ax.set_xlabel("Số sao")
    ax.set_ylabel("Tỉ lệ chê (%)")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, 100)
    ax.legend(title="Khía cạnh")


def ve_diem_khia_canh(df_absa, ax):
    df_ro_nganh = df_absa[df_absa['nganh_hang'] != NGANH_KHONG_RO]
    bang = pd.DataFrame()
    for ma in DS_KHIA_CANH:
        df_co_nhac = df_ro_nganh[df_ro_nganh['cx_' + ma].notna()].copy()
        df_co_nhac['la_khen'] = df_co_nhac['cx_' + ma] == NHAN_TICH_CUC
        bang[TEN_KHIA_CANH_ABSA[ma]] = df_co_nhac.groupby('nganh_hang')['la_khen'].mean() * 10

    bang.T.plot(kind='bar', rot=15, width=0.8, ax=ax)
    ax.axhline(5, color='black', linestyle='--')
    ax.set_title("Điểm từng khía cạnh theo ngành hàng (thang 10)")
    ax.set_xlabel("Khía cạnh")
    ax.set_ylabel("Điểm")
    ax.set_ylim(0, 10)
    ax.legend(title="Ngành hàng", ncol=4, loc='upper center')


## ==================== PHẦN F - WEB ====================

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
for ten_nhom, ds_bieu_do in DS_NHOM_BIEU_DO:
    for ten_file, chu_thich in ds_bieu_do:
        CHU_THICH_BIEU_DO[ten_file] = chu_thich

DS_ANH_BIEU_DO = list(CHU_THICH_BIEU_DO)
for ten_file in DS_BIEU_DO_DASHBOARD:
    DS_ANH_BIEU_DO.append(TIEN_TO_ANH_NHO + ten_file)


## 2. Vẽ biểu đồ

def so_sanh_mo_hinh() -> tuple:
    x_data = df['noi_dung_sach']
    y_data = df['label']
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                        test_size=TI_LE_TEST,
                                                        random_state=RANDOM_STATE,
                                                        stratify=y_data)
    pipe_nb = tao_pipeline_phan_loai(MultinomialNB())
    pipe_nb.fit(x_train, y_train)
    pipe_lr = tao_pipeline_phan_loai(LogisticRegression(max_iter=MAX_ITER, class_weight='balanced'))
    pipe_lr.fit(x_train, y_train)
    pipe_svc = tao_pipeline_phan_loai(LinearSVC(class_weight='balanced'))
    pipe_svc.fit(x_train, y_train)

    ds_ket_qua = []
    ds_ket_qua.append(danh_gia_mo_hinh('Naive Bayes', pipe_nb, x_train, y_train, x_test, y_test))
    ds_ket_qua.append(danh_gia_mo_hinh('Logistic Regression', pipe_lr, x_train, y_train, x_test, y_test))
    ds_ket_qua.append(danh_gia_mo_hinh('Linear SVC', pipe_svc, x_train, y_train, x_test, y_test))

    ds_nhan = [NHAN_TIEU_CUC, NHAN_TICH_CUC]
    ma_tran_lr = confusion_matrix(y_test, pipe_lr.predict(x_test), labels=ds_nhan)
    return pd.DataFrame(ds_ket_qua), ma_tran_lr


def ve_bieu_do():
    ax = tao_khung(7, 4.5)
    ve_phan_bo_cam_xuc(df, ax)
    luu_bieu_do('01_phan_bo_cam_xuc.png')

    ax = tao_khung(7, 4.5)
    ve_phan_bo_so_sao(df, ax)
    luu_bieu_do('02_phan_bo_so_sao.png')

    ax = tao_khung(7, 4.5)
    ve_cam_xuc_theo_sao(df, ax)
    luu_bieu_do('03_cam_xuc_theo_sao.png')

    ax = tao_khung(7, 4.5)
    ve_ti_le_che_theo_nganh(df, ax)
    luu_bieu_do('04_ti_le_che_theo_nganh.png')

    bang_so_sanh, ma_tran_lr = so_sanh_mo_hinh()
    ax = tao_khung(8, 4.5)
    ve_so_sanh_mo_hinh(bang_so_sanh, ax)
    luu_bieu_do('05_so_sanh_mo_hinh.png')

    ax = tao_khung(6, 4.5)
    ve_ma_tran_nham_lan(ma_tran_lr, ax)
    luu_bieu_do('06_ma_tran_nham_lan.png')

    bang_tu = lay_tu_quan_trong(mo_hinh, SO_TU_QUAN_TRONG)
    ax = tao_khung(8, 7)
    ve_tu_quan_trong(bang_tu, ax)
    luu_bieu_do('07_tu_quan_trong.png')

    ax = tao_khung(8, 5)
    ve_khen_che_khia_canh(df_absa, ax)
    luu_bieu_do('08_khen_che_khia_canh.png')

    ax = tao_khung(9, 4.5)
    ve_ti_le_che_khia_canh_nganh(df_absa, ax)
    luu_bieu_do('09_ti_le_che_khia_canh_nganh.png')

    ax = tao_khung(8, 5)
    ve_ti_le_che_theo_sao(df_absa, ax)
    luu_bieu_do('10_ti_le_che_theo_sao.png')

    ax = tao_khung(10, 5)
    ve_diem_khia_canh(df_absa, ax)
    luu_bieu_do('11_diem_khia_canh.png')

    bang_f1 = pd.read_csv('../dataset/ket_qua_f1_khia_canh.csv')
    ax = tao_khung(8, 5)
    ve_f1_khia_canh(bang_f1, ax)
    luu_bieu_do('12_f1_khia_canh.png')

    ax = tao_khung_nho()
    ve_khen_che_khia_canh(df_absa, ax)
    ax.set_title("Số bình luận khen và chê theo khía cạnh")
    ax.set_ylabel("")
    ax.legend(loc='upper right')
    thu_nho_so_tren_cot(ax)
    luu_bieu_do(TIEN_TO_ANH_NHO + '08_khen_che_khia_canh.png')

    ax = tao_khung_nho()
    ve_ti_le_che_khia_canh_nganh(df_absa, ax)
    ax.set_title("Tỉ lệ chê (%) theo khía cạnh và ngành hàng")
    ax.set_xlabel("")
    ax.set_ylabel("")
    for nhan in ax.get_xticklabels():
        nhan.set_horizontalalignment('right')
    luu_bieu_do(TIEN_TO_ANH_NHO + '09_ti_le_che_khia_canh_nganh.png')

    ax = tao_khung_nho()
    ve_ti_le_che_theo_sao(df_absa, ax)
    ax.set_title("Tỉ lệ chê của từng khía cạnh theo số sao")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    luu_bieu_do(TIEN_TO_ANH_NHO + '10_ti_le_che_theo_sao.png')

    ax = tao_khung_nho()
    ve_phan_bo_so_sao(df, ax)
    ax.margins(y=0.15)
    luu_bieu_do(TIEN_TO_ANH_NHO + '02_phan_bo_so_sao.png')

    ax = tao_khung_nho()
    ve_cam_xuc_theo_sao(df, ax)
    ax.set_ylim(0, 120)
    ax.legend(title="Cảm xúc", loc='center left', bbox_to_anchor=(1, 0.5))
    thu_nho_so_tren_cot(ax)
    luu_bieu_do(TIEN_TO_ANH_NHO + '03_cam_xuc_theo_sao.png')

    ax = tao_khung_nho()
    ve_ti_le_che_theo_nganh(df, ax)
    ax.set_ylabel("")
    luu_bieu_do(TIEN_TO_ANH_NHO + '04_ti_le_che_theo_nganh.png')

    ax = tao_khung_nho()
    ve_so_sanh_mo_hinh(bang_so_sanh, ax)
    ax.set_ylabel("")
    luu_bieu_do(TIEN_TO_ANH_NHO + '05_so_sanh_mo_hinh.png')

    bang_tu = lay_tu_quan_trong(mo_hinh, SO_TU_QUAN_TRONG_NHO)
    ax = tao_khung_nho()
    ve_tu_quan_trong(bang_tu, ax)
    ax.set_xlabel("Hệ số của mô hình (âm: tiêu cực, dương: tích cực)")
    ax.set_ylabel("")
    ax.tick_params(axis='y', labelsize=CO_SO_TREN_COT_NHO)
    luu_bieu_do(TIEN_TO_ANH_NHO + '07_tu_quan_trong.png')

    ax = tao_khung_nho()
    ve_f1_khia_canh(bang_f1, ax)
    ax.set_title("F1 nhận diện từng khía cạnh trên các câu gán tay")
    ax.set_ylabel("")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim(0, 1.15)
    thu_nho_so_tren_cot(ax)
    luu_bieu_do(TIEN_TO_ANH_NHO + '12_f1_khia_canh.png')

    print("Đã lưu", len(DS_ANH_BIEU_DO), "ảnh biểu đồ vào thư mục bieu_do/")


## 3. Kiểm tra đầu vào và dự đoán

def kiem_tra_binh_luan(binh_luan) -> str:
    if not isinstance(binh_luan, str):
        return "Bình luận phải là một đoạn chữ."
    if binh_luan.strip() == '':
        return "Bạn chưa nhập bình luận."
    if len(binh_luan) > DO_DAI_BINH_LUAN_TOI_DA:
        return "Bình luận dài quá " + str(DO_DAI_BINH_LUAN_TOI_DA) + " ký tự."
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

    ket_qua = du_doan(binh_luan, mo_hinh)
    noi_dung_sach = ket_qua['noi_dung_da_lam_sach']
    if noi_dung_sach == '':
        return 400, {'loi': "Bình luận không có chữ nào để phân tích."}

    ket_qua['tu_anh_huong'] = lay_tu_anh_huong(mo_hinh, noi_dung_sach, SO_TU_ANH_HUONG)

    ket_qua['so_sao'] = so_sao
    if so_sao is None:
        ket_qua['doi_chieu_sao'] = ''
    else:
        ket_qua['doi_chieu_sao'] = doi_chieu_sao(so_sao, ket_qua['nhan'])
    ket_qua['khia_canh'] = lay_khia_canh(binh_luan)
    return 200, ket_qua


def lay_khia_canh(binh_luan: str) -> list:
    ket_qua_absa = phan_tich_absa(binh_luan, mo_hinh_cam_xuc_absa)
    ds_khia_canh = []
    for khia_canh in ket_qua_absa['aspects']:
        if khia_canh['sentiment'] == NHAN_TICH_CUC:
            cam_xuc = TEN_TICH_CUC
        else:
            cam_xuc = TEN_TIEU_CUC
        ds_khia_canh.append({'ten': TEN_KHIA_CANH_ABSA[khia_canh['aspect']],
                             'cam_xuc': cam_xuc,
                             'xac_suat_tich_cuc': khia_canh['xac_suat_tich_cuc']})
    return ds_khia_canh


def lay_du_lieu_dashboard() -> dict:
    ds_bieu_do = []
    for ten_file in DS_BIEU_DO_DASHBOARD:
        ds_bieu_do.append({'ten_file': ten_file,
                           'ten_file_nho': TIEN_TO_ANH_NHO + ten_file,
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

        so_byte_toi_da = DO_DAI_BINH_LUAN_TOI_DA * 4 + 1000
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

    may_chu = HTTPServer(('127.0.0.1', CONG_WEB), XuLyYeuCau)
    print("Web đang chạy tại: http://localhost:" + str(CONG_WEB))
    print("Bấm Ctrl + C để dừng.")
    try:
        may_chu.serve_forever()
    except KeyboardInterrupt:
        print("Đã dừng web.")
