#ここまでは完璧20241125
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageOps, ExifTags
import numpy as np
import sys


# ======================================
# Streamlitを用いた画像入力部分
# ======================================
st.title("配線検出アプリ")
st.write("画像をアップロードするか、カメラで撮影してください。")

uploaded_image = st.file_uploader("画像をアップロードする", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("カメラで撮影する")

if uploaded_image is not None:
    input_image = Image.open(uploaded_image)
elif camera_image is not None:
    input_image = Image.open(camera_image)
else:
    st.warning("画像をアップロードするか、カメラで撮影してください。")
    st.stop()


# ======================================
# 関数: 座標の上下端を取得する関数
# ======================================
def topbottom(img):
    img = (img > 128) * 255
    rows, cols = img.shape

    # 上端座標を取得
    for i in range(rows):
        for j in range(cols - 1, -1, -1):
            if img[i, j] == 255:
                img_top = (i, j)
                break
        if 'img_top' in locals():
            break

    # 下端座標を取得
    for i in range(rows - 1, -1, -1):
        for j in range(cols):
            if img[i, j] == 255:
                img_bottom = (i, j)
                break
        if 'img_bottom' in locals():
            break

    return img_top, img_bottom


# ======================================
# 関数: 画像を処理する関数
# ======================================
def image_press(img):
    # EXIF情報から画像の向きを取得し、必要に応じて回転させる
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(img._getexif().items())

        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # EXIF情報がない場合はスキップ
        pass

    # アスペクト比を維持し正方形化
    img_new = ImageOps.pad(img, (max(img.size), max(img.size)), color=(255, 255, 255))

    # サイズを416x416にリサイズ
    img_new = img_new.resize((416, 416))

    return img_new


# ======================================
# モデルの読み込みと予測
# ======================================
model = YOLO("e_meter_segadd2.pt")
org_img = image_press(input_image)

# YOLOモデルで画像を予測（クラス0: line）
results = model.predict(org_img, imgsz=416, conf=0.5, classes=0)

# 配線検出エラーチェック
if results[0].masks is None:
    st.error("配線の検出に失敗しました。目視で確認してください。")
    st.stop()
if len(results[0].masks) < 3:
    for r in results:
        im_array = r.plot(boxes=False)
        im = Image.fromarray(im_array[..., ::-1])
    st.image(im, caption="検出結果（不十分）", use_column_width=True)
    st.error("配線の検出が不十分です。目視で確認してください。")
    st.stop()


# ======================================
# マスク画像の処理
# ======================================
processed_data = {'images': [], 'coordinates': [], 'classification': ''}

# マスク画像ごとに処理
for i, r in enumerate(results[0].masks):
    mask_img = r.data[0].cpu().numpy() * 255
    mask_img = mask_img.astype(int)

    # 上下端の座標を取得
    top, bottom = topbottom(mask_img)

    # 結果を保存
    processed_data['images'].append(mask_img)
    processed_data['coordinates'].append((top, bottom))


# ======================================
# 座標の整理
# ======================================
coordinates_list = processed_data['coordinates']
connect_list = np.array(coordinates_list)

top_list = [coord[0] for coord in connect_list]
bottom_list = [coord[1] for coord in connect_list]

top_list = np.array(top_list)
bottom_list = np.array(bottom_list)


# ======================================
# 配線が3本の場合の補完処理
# ======================================
if len(top_list) == 3:
    y0, x0 = zip(*sorted(top_list, key=lambda x: x[1]))

    xl1 = x0[1] - x0[0]
    xl2 = x0[2] - x0[1]

    if xl1 < xl2:
        width = xl1
        topx_dummy = x0[2] - width
    else:
        width = xl2
        topx_dummy = x0[0] + width

    y_avr = int(sum(y0) / len(y0))
    top_list = np.append(top_list, [(y_avr, topx_dummy)], axis=0)

    sorted_comptop = np.array(sorted(top_list, key=lambda x: x[1]))
    connect_center = sorted_comptop[0][1] + width * 2

    k, l = [], []
    y0, x0 = zip(*sorted(bottom_list, key=lambda x: x[1]))
    y_avr = int(sum(y0) / len(y0))

    for line in x0:
        if line < connect_center:
            k.append(line)
        else:
            l.append(line)

    if len(k) == 1:
        btmx_dummy = k[0] - 5
    if len(l) == 1:
        btmx_dummy = l[0] + 5
    bottom_list = np.append(bottom_list, [(y_avr, btmx_dummy)], axis=0)


# ======================================
# 座標のソートと分類
# ======================================
sorted_top = np.array(sorted(top_list, key=lambda x: x[1]))
sorted_bottom = np.array([tup for _, tup in sorted(zip(top_list, bottom_list), key=lambda x: x[0][1])])

center = sum([coords[1] for coords in sorted_bottom]) / 4

if np.all(sorted_bottom[::2, 1] < center) and np.all(sorted_bottom[1::2, 1] > center):
    processed_data['classification'] = "正結線の可能性が高いです"
else:
    processed_data['classification'] = "誤結線の可能性があります。"

st.write(f"分類結果: {processed_data['classification']}")


# ======================================
# 検出結果の描画と表示
# ======================================
for r in results:
    im_array = r.plot(boxes=False)
    im = Image.fromarray(im_array[..., ::-1])

draw = ImageDraw.Draw(im)

for i, (y, x) in enumerate(sorted_top):
    x1, y1, x2, y2 = x - 5, y - 5, x + 5, y + 5
    fill_color = (255, 0, 255) if i < 2 else (0, 0, 255)
    draw.ellipse((x1, y1, x2, y2), fill=fill_color)

for i, (y, x) in enumerate(sorted_bottom):
    x1, y1, x2, y2 = x - 5, y - 5, x + 5, y + 5
    fill_color = (255, 0, 255) if i < 2 else (0, 0, 255)
    draw.ellipse((x1, y1, x2, y2), fill=fill_color)

st.image(im, caption="検出結果", use_column_width=True)
