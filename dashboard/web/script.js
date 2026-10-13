const DS_CAU_MAU = [
    'Hàng đẹp, giao nhanh, shop nhiệt tình',
    'Giao sai màu, nhắn shop ko trả lời',
    'Son lên màu đẹp, mỗi tội hơi khô môi',
    'Sp tốt nhưng giao hàng hơi lâu',
    'Bao bì:đẹp Mùi hương:hắc  Giao nhanh, giá rẻ nhưng shop trả lời chậm',
    'Bình thường, không có gì đặc biệt',
];

const LOP_THEO_NHAN = {
    'Tích cực': 'tich-cuc',
    'Trung lập': 'trung-lap',
    'Tiêu cực': 'tieu-cuc',
};

const DOI_CHIEU_SAO = {
    'Khớp': { lop: 'khop', nhanXet: 'khớp với' },
    'Lệch nhẹ': { lop: 'lech-nhe', nhanXet: 'hơi lệch so với' },
    'Lệch sao': { lop: 'lech-sao', nhanXet: 'trái ngược với' },
};

let soSaoDaChon = null;

const oBinhLuan = document.getElementById('o-binh-luan');
const soKyTu = document.getElementById('so-ky-tu');
const nutDuDoan = document.getElementById('nut-du-doan');
const nutXoa = document.getElementById('nut-xoa');
const thongBaoLoi = document.getElementById('thong-bao-loi');
const choKetQua = document.getElementById('cho-ket-qua');
const noiDungKetQua = document.getElementById('noi-dung-ket-qua');
const khungLichSu = document.getElementById('khung-lich-su');

function doiSangPhanTram(p) {
    return (p * 100).toFixed(2).replace('.', ',') + '%';
}

function capNhatSoKyTu() {
    soKyTu.textContent = oBinhLuan.value.length + ' / ' + oBinhLuan.maxLength + ' ký tự';
}

function capNhatSao() {
    const dsNutSao = document.querySelectorAll('#chon-sao button');
    dsNutSao.forEach(function (nut, viTri) {
        const duocChon = soSaoDaChon !== null && viTri < soSaoDaChon;
        nut.classList.toggle('da-chon', duocChon);
        nut.setAttribute('aria-pressed', soSaoDaChon === viTri + 1 ? 'true' : 'false');
    });
    document.getElementById('chu-so-sao').textContent =
        soSaoDaChon === null ? 'Chưa chọn' : soSaoDaChon + ' sao (bấm lại để bỏ chọn)';
}

function hienDoiChieuSao(ketQua) {
    const dong = document.getElementById('doi-chieu-sao');
    if (ketQua.so_sao === null) {
        dong.hidden = true;
        return;
    }
    const muc = DOI_CHIEU_SAO[ketQua.doi_chieu_sao];
    dong.textContent = ketQua.so_sao + ' sao ' + muc.nhanXet + ' nội dung ' + ketQua.nhan.toLowerCase() + '.';
    dong.className = 'doi-chieu doi-chieu-' + muc.lop;
    dong.hidden = false;
}

function hienLoi(cauBaoLoi) {
    thongBaoLoi.textContent = cauBaoLoi;
    thongBaoLoi.hidden = false;
}

function taoO(noiDung, tenLop) {
    const o = document.createElement('td');
    o.textContent = noiDung;
    if (tenLop) {
        o.className = tenLop;
    }
    return o;
}

function datKhungKetQua(coKetQua) {
    choKetQua.hidden = coKetQua;
    noiDungKetQua.hidden = !coKetQua;
}

function hienKetQua(ketQua) {
    const nhan = document.getElementById('nhan');
    nhan.textContent = ketQua.nhan;
    nhan.className = 'nhan nhan-' + LOP_THEO_NHAN[ketQua.nhan];

    hienDoiChieuSao(ketQua);

    const thanBangKhiaCanh = document.getElementById('than-bang-khia-canh');
    thanBangKhiaCanh.replaceChildren();
    for (const muc of ketQua.khia_canh) {
        const dong = document.createElement('tr');
        dong.appendChild(taoO(muc.ten));
        dong.appendChild(taoO(muc.cam_xuc, 'chu-' + LOP_THEO_NHAN[muc.cam_xuc]));
        dong.appendChild(taoO(doiSangPhanTram(muc.xac_suat_tich_cuc), 'cot-so'));
        thanBangKhiaCanh.appendChild(dong);
    }
    const khongCoKhiaCanh = ketQua.khia_canh.length === 0;
    document.getElementById('khong-co-khia-canh').hidden = !khongCoKhiaCanh;
    document.getElementById('bang-khia-canh').hidden = khongCoKhiaCanh;

    document.getElementById('noi-dung-sach').textContent = ketQua.noi_dung_da_lam_sach;

    const thanBangTu = document.getElementById('than-bang-tu');
    thanBangTu.replaceChildren();
    for (const muc of ketQua.tu_anh_huong) {
        const dong = document.createElement('tr');
        const laTichCuc = muc.dong_gop > 0;
        dong.appendChild(taoO(muc.tu));
        dong.appendChild(taoO(laTichCuc ? 'Tích cực' : 'Tiêu cực', laTichCuc ? 'chu-tich-cuc' : 'chu-tieu-cuc'));
        dong.appendChild(taoO((laTichCuc ? '+' : '') + muc.dong_gop.toFixed(4).replace('.', ','), 'cot-so'));
        thanBangTu.appendChild(dong);
    }
    const khongCoTu = ketQua.tu_anh_huong.length === 0;
    document.getElementById('khong-co-tu').hidden = !khongCoTu;
    document.getElementById('bang-tu').hidden = khongCoTu;

    datKhungKetQua(true);
}

function themVaoLichSu(binhLuan, ketQua) {
    const dong = document.createElement('tr');
    dong.appendChild(taoO(binhLuan));
    dong.appendChild(taoO(ketQua.so_sao === null ? '–' : ketQua.so_sao + ' ★', 'cot-so'));
    dong.appendChild(taoO(ketQua.nhan, 'chu-' + LOP_THEO_NHAN[ketQua.nhan]));
    dong.appendChild(taoO(doiSangPhanTram(ketQua.xac_suat_tich_cuc), 'cot-so'));
    dong.appendChild(taoO(ketQua.so_sao === null ? '–' : ketQua.doi_chieu_sao));
    document.getElementById('than-bang-lich-su').prepend(dong);
    khungLichSu.hidden = false;
}

async function duDoan() {
    const binhLuan = oBinhLuan.value;
    thongBaoLoi.hidden = true;
    if (binhLuan.trim() === '') {
        hienLoi('Bạn chưa nhập bình luận.');
        return;
    }

    nutDuDoan.disabled = true;
    try {
        const traLoi = await fetch('/api/du-doan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ binh_luan: binhLuan, so_sao: soSaoDaChon }),
        });
        const ketQua = await traLoi.json();
        if (!traLoi.ok) {
            hienLoi(ketQua.loi);
            return;
        }
        hienKetQua(ketQua);
        themVaoLichSu(binhLuan, ketQua);
    } catch (loi) {
        hienLoi('Không kết nối được với web. Kiểm tra lại file dashboard.py còn đang chạy không.');
    } finally {
        nutDuDoan.disabled = false;
    }
}

let daTaiDashboard = false;

const khungPhongTo = document.getElementById('khung-phong-to');

function phongToBieuDo(bieuDo) {
    const anh = document.getElementById('anh-phong-to');
    anh.src = '/bieu_do/' + bieuDo.ten_file;
    anh.alt = bieuDo.chu_thich;
    document.getElementById('chu-thich-phong-to').textContent = bieuDo.chu_thich;
    khungPhongTo.showModal();
}

function hienDashboard(duLieu) {
    const luoi = document.getElementById('luoi-bieu-do');
    luoi.replaceChildren();
    for (const bieuDo of duLieu.ds_bieu_do) {
        const khung = document.createElement('figure');
        khung.className = 'khung-bieu-do';
        khung.tabIndex = 0;
        khung.title = bieuDo.chu_thich + ' (bấm để phóng to)';
        const anh = document.createElement('img');
        anh.src = '/bieu_do/' + bieuDo.ten_file_nho;
        anh.alt = bieuDo.chu_thich;
        khung.appendChild(anh);
        khung.addEventListener('click', function () {
            phongToBieuDo(bieuDo);
        });
        khung.addEventListener('keydown', function (suKien) {
            if (suKien.key === 'Enter') {
                phongToBieuDo(bieuDo);
            }
        });
        luoi.appendChild(khung);
    }
}

khungPhongTo.addEventListener('click', function () {
    khungPhongTo.close();
});

async function taiDashboard() {
    const loiDashboard = document.getElementById('loi-dashboard');
    loiDashboard.hidden = true;
    try {
        const traLoi = await fetch('/api/dashboard');
        hienDashboard(await traLoi.json());
        daTaiDashboard = true;
    } catch (loi) {
        loiDashboard.textContent = 'Không tải được dashboard. Kiểm tra lại file dashboard.py còn đang chạy không.';
        loiDashboard.hidden = false;
    }
}

function chonTrang(idTrang) {
    document.querySelectorAll('.trang').forEach(function (trang) {
        trang.hidden = trang.id !== idTrang;
    });
    document.querySelectorAll('.muc-menu').forEach(function (muc) {
        const dangChon = muc.dataset.trang === idTrang;
        muc.classList.toggle('dang-chon', dangChon);
        muc.setAttribute('aria-current', dangChon ? 'page' : 'false');
    });
    if (idTrang === 'trang-dashboard' && !daTaiDashboard) {
        taiDashboard();
    }
}

document.querySelectorAll('.muc-menu').forEach(function (muc) {
    muc.addEventListener('click', function () {
        window.location.hash = muc.dataset.trang === 'trang-dashboard' ? 'dashboard' : '';
        chonTrang(muc.dataset.trang);
    });
});
chonTrang(window.location.hash === '#dashboard' ? 'trang-dashboard' : 'trang-phan-tich');

for (const cauMau of DS_CAU_MAU) {
    const nut = document.createElement('button');
    nut.type = 'button';
    nut.textContent = cauMau;
    nut.title = cauMau;
    nut.addEventListener('click', function () {
        oBinhLuan.value = cauMau;
        capNhatSoKyTu();
        duDoan();
    });
    document.getElementById('ds-mau').appendChild(nut);
}

for (let soSao = 1; soSao <= 5; soSao++) {
    const nut = document.createElement('button');
    nut.type = 'button';
    nut.textContent = '★';
    nut.setAttribute('aria-label', soSao + ' sao');
    nut.addEventListener('click', function () {
        soSaoDaChon = soSaoDaChon === soSao ? null : soSao;
        capNhatSao();
    });
    document.getElementById('chon-sao').appendChild(nut);
}
capNhatSao();

oBinhLuan.addEventListener('input', capNhatSoKyTu);
oBinhLuan.addEventListener('keydown', function (suKien) {
    if (suKien.key === 'Enter' && (suKien.ctrlKey || suKien.metaKey)) {
        duDoan();
    }
});
nutDuDoan.addEventListener('click', duDoan);
nutXoa.addEventListener('click', function () {
    oBinhLuan.value = '';
    soSaoDaChon = null;
    capNhatSao();
    capNhatSoKyTu();
    thongBaoLoi.hidden = true;
    datKhungKetQua(false);
    oBinhLuan.focus();
});
