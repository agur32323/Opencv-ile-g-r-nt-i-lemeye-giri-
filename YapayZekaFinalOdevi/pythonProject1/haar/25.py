
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
import cap
import cv2
import numpy as np
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
import tkinter as tk

def yuz_tanima_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Hata", "Web kamerası açılamıyor.")
        return
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if ret:
            gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yuz_casc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            yuzler = yuz_casc.detectMultiScale(gri, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in yuzler:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow('Yuz Tanima', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

def yuz_bulaniklastir_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Hata", "Web kamerası açılamıyor.")
        return

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if ret:
            gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yuz_casc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            yuzler = yuz_casc.detectMultiScale(gri, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in yuzler:
                yuz = frame[y:y+h, x:x+w]
                yuz = cv2.GaussianBlur(yuz, (99, 99), 30)
                frame[y:y+h, x:x+w] = yuz

            cv2.imshow('Yuz Bulaniklastirma', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
def yuz_ve_goz_tanima_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Hata", "Web kamerası açılamıyor.")
        return
    yuz_casc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    goz_casc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if ret:
            gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yuzler = yuz_casc.detectMultiScale(gri, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in yuzler:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gri = gri[y:y + h, x:x + w]
                roi_renkli = frame[y:y + h, x:x + w]
                gozler = goz_casc.detectMultiScale(roi_gri)
                for (ex, ey, ew, eh) in gozler:
                    cv2.rectangle(roi_renkli, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            cv2.imshow('Yuz ve Goz Tanima', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

def yuz_tanima():
        dosya_yolu = filedialog.askopenfilename(title="Bir fotoğraf seçin",
                                                filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not dosya_yolu:  # Kullanıcı iptal ettiyse
            return

        image = cv2.imread(dosya_yolu)
        gri = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        yuz_casc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        yuzler = yuz_casc.detectMultiScale(gri, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in yuzler:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Fotoğrafta Yüz Tanıma', image)

        cv2.waitKey(0)


cv2.destroyAllWindows()


def yuz_ve_goz_tanima_fotograf():
            dosya_yolu = filedialog.askopenfilename(title="Bir fotoğraf seçin",
                                                    filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
            if not dosya_yolu:  # Kullanıcı iptal ettiyse
                return

            image = cv2.imread(dosya_yolu)
            gri = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            yuz_casc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            goz_casc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

            yuzler = yuz_casc.detectMultiScale(gri, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in yuzler:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_gri = gri[y:y + h, x:x + w]
                gozler = goz_casc.detectMultiScale(roi_gri)
                for (ex, ey, ew, eh) in gozler:
                    cv2.rectangle(image[y:y + h, x:x + w], (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

            cv2.imshow('Fotoğrafta Yüz ve Göz Tanıma', image)
            cv2.waitKey(0)

            cv2.destroyAllWindows()
def fotograf_grilestir():
    dosya_yolu = filedialog.askopenfilename(title="Bir fotoğraf seçin", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if not dosya_yolu:
        return

    image = cv2.imread(dosya_yolu)
    gri_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    cv2.imshow('Orjinal resim', image)
    cv2.imshow('Grileştirilmiş Fotoğraf', gri_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def webcam_grilestir():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Webcam Grayscale', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def fotograf_adaptive_threshold():
    dosya_yolu = filedialog.askopenfilename(title="Bir fotoğraf seçin", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if not dosya_yolu:
        return

    image = cv2.imread(dosya_yolu, 0)

    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imshow('Orjinal resim', image)
    cv2.imshow('Adaptive Threshold', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def webcam_adaptive_threshold():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        thresh_frame = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        cv2.imshow('Webcam Adaptive Threshold', thresh_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
def fotograf_otsu_esikleme():
    dosya_yolu = filedialog.askopenfilename(title="Bir fotoğraf seçin",
                                            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if not dosya_yolu:
        return

    image_gray = cv2.imread(dosya_yolu, cv2.IMREAD_GRAYSCALE)


    _, otsu_thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow('Orjinal resim', image_gray)
    cv2.imshow('Otsu Thresholding', otsu_thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def webcam_otsu_esikleme():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, otsu_thresh_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cv2.imshow('Webcam Otsu Thresholding', otsu_thresh_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
def fotograf_kenarlik_ekleme():
    dosya_yolu = filedialog.askopenfilename(title="Bir fotoğraf seçin",
                                            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if not dosya_yolu:
        return

    image = cv2.imread(dosya_yolu)
    kenarlik_genisligi = 10
    kenarlik_rengi = (0, 0, 255)
    image_with_border = cv2.copyMakeBorder(image, kenarlik_genisligi, kenarlik_genisligi, kenarlik_genisligi, kenarlik_genisligi, cv2.BORDER_CONSTANT, value=kenarlik_rengi)

    cv2.imshow('orjinal resim',image)
    cv2.imshow('Fotoğraf Kenarlık Ekleme', image_with_border)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def fotograf_keskinlestirme():
    dosya_yolu = filedialog.askopenfilename(title="Bir fotoğraf seçin",
                                            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if not dosya_yolu:
        return
    image = cv2.imread(dosya_yolu)

    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])

    sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)

    cv2.imshow('orjinal resim',image)

    cv2.imshow('Fotoğraf Keskinleştirme', sharpened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def webcam_keskinlestirme():
    cap = cv2.VideoCapture(0)

    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        sharpened_frame = cv2.filter2D(frame, -1, sharpening_kernel)
        cv2.imshow('Webcam Keskinleştirme', sharpened_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
resim = None

def uygula_keskinlestirme(val):
    global resim
    if resim is None:
        return
    alpha = float(val)
    kernel = np.array([[0, -alpha, 0],
                       [-alpha, 1 + 4 * alpha, -alpha],
                       [0, -alpha, 0]])

    sharp_image = cv2.filter2D(src=resim, ddepth=-1, kernel=kernel)
    cv2.imshow("Keskinlestirilmis Resim", sharp_image)

def resim_sec():
    global resim
    file_path = filedialog.askopenfilename()
    if file_path:
        resim = cv2.imread(file_path, cv2.IMREAD_COLOR)
        cv2.imshow("Orijinal Resim", resim)


def fotograf_bulaniklastirma():


    dosya_yolu = filedialog.askopenfilename(title="Bir fotoğraf seçin",
                                            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if not dosya_yolu:
        return


    image = cv2.imread(dosya_yolu)


    blur_amount = bulaniklik_derecesi.get()
    blur_amount = blur_amount + 1 if blur_amount % 2 == 0 else blur_amount
    blurred_image = cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)

    blur_amount = max(1, blur_amount)
    blurred_image = cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)


    cv2.imshow('Fotoğraf Bulanıklaştırma', blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ayarla_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def fotograf_gamma_filtreleme():
    dosya_yolu = filedialog.askopenfilename(title="Bir fotoğraf seçin",
                                            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if not dosya_yolu:
        return

    image = cv2.imread(dosya_yolu)

    gamma = float(simpledialog.askstring("Gamma Düzeltmesi", "Gamma değerini girin:", initialvalue="1.0"))

    gamma_filtered_image = ayarla_gamma(image, gamma)

    cv2.imshow('Gamma Filtrelenmiş Fotoğraf', gamma_filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def fotograf_histogram_goster():
    dosya_yolu = filedialog.askopenfilename(title="Bir fotoğraf seçin",
                                            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if not dosya_yolu:
        return

    image = cv2.imread(dosya_yolu)
    for i, color in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    plt.title('Histogram')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')

    plt.show()

def canny_kenar_tespiti():
    dosya_yolu = filedialog.askopenfilename(title="Bir fotoğraf seçin",
                                            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if not dosya_yolu:
        return

    image = cv2.imread(dosya_yolu, cv2.IMREAD_GRAYSCALE)

    edges = cv2.Canny(image, 100, 200)

    cv2.imshow('Canny Kenar Tespiti', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sobel_kenar_tespiti(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


    sobelx = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    sobely = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)

    sobelx = np.uint8(np.absolute(sobelx))
    sobely = np.uint8(np.absolute(sobely))


    sobel_combined = cv2.bitwise_or(sobelx, sobely)

    return sobel_combined


def open_image_and_detect_edges():
    file_path = filedialog.askopenfilename()
    if file_path:
        edges = sobel_kenar_tespiti(file_path)

        cv2.imshow('Sobel kenar tespiti', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



def harris_köşe_tespiti(image_path):

    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)

    dst = cv2.dilate(dst, None)

    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    return img



def open_image_and_detect_corners():
    file_path = filedialog.askopenfilename()
    if file_path:
        corners = harris_köşe_tespiti(file_path)

        cv2.imshow('Harris Corner Detection', corners)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



def contour_detection(image_path):

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)


    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)

    return contour_img


def open_image_and_detect_contours():
    file_path = filedialog.askopenfilename()
    if file_path:
        contours_img = contour_detection(file_path)
        cv2.imshow('Contour Detection', contours_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def watershed_algorithm(image_path):

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)


    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1

    markers[unknown == 255] = 0


    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    return img


def open_image_and_apply_watershed():
    file_path = filedialog.askopenfilename()
    if file_path:
        result_img = watershed_algorithm(file_path)

        cv2.imshow('Watershed Algorithm Result', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



root = tk.Tk()
root.title('Webcam ile Yüz Tanıma & Bulanıklaştırma')
root.geometry("800x600")

tanima_buton = tk.Button(root, text="Webcam Yüz Tanıma", command=yuz_tanima_webcam)
tanima_buton.grid(row=75, column=0)

bulanik_buton = tk.Button(root, text="Webcam Yüz Bulanıklaştır", command=yuz_bulaniklastir_webcam)
bulanik_buton.grid(row=74, column=0)

tanima_buton = tk.Button(root, text="Webcam Yüz ve Göz Tanıma", command=yuz_ve_goz_tanima_webcam)
tanima_buton.grid(row=73, column=0)

fotograf_tanima_buton = tk.Button(root, text="Fotoğrafta Yüz Tanıma",command=yuz_tanima)
fotograf_tanima_buton.grid(row=72, column=0)

fotograf_yuz_goz_tanima_buton = tk.Button(root, text="Fotoğrafta Yüz ve Göz Tanıma", command=yuz_ve_goz_tanima_fotograf)
fotograf_yuz_goz_tanima_buton.grid(row=60, column=90)

fotograf_grilestir_buton = tk.Button(root, text="Fotoğrafı Grileştir", command=fotograf_grilestir)
fotograf_grilestir_buton.grid(row=71, column=0)

webcam_grilestir_buton = tk.Button(root, text="Webcam Görüntüsünü Grileştir", command=webcam_grilestir)
webcam_grilestir_buton.grid(row=70, column=0)

sec_btn = tk.Button(root, text="Keskinleştirme Yapılacak Resmi Seç", command=resim_sec)
sec_btn.grid(row=150, column=90)
sharpness_scale = tk.Scale(root, to=10, from_=1, resolution=0.1, orient='horizontal', label='Keskinlestirme Faktoru', command=uygula_keskinlestirme)
sharpness_scale.set(1)
sharpness_scale.grid(row=160, column=90)
kontrol_frame = tk.Frame(root)
kontrol_frame.grid(row=170, column=90)

adaptive_thresh_buton = tk.Button(root, text="Fotoğrafa Adaptive Threshold Uygula", command=fotograf_adaptive_threshold)
adaptive_thresh_buton.grid(row=69, column=0)

webcam_adaptive_thresh_buton = tk.Button(root, text="Webcam Görüntüsüne Adaptive Threshold Uygula", command=webcam_adaptive_threshold)
webcam_adaptive_thresh_buton.grid(row=50, column=90)

otsu_esikleme_buton = tk.Button(root, text="Fotoğrafa Otsu Eşikleme Uygula", command=fotograf_otsu_esikleme)
otsu_esikleme_buton.grid(row=65, column=0)

webcam_otsu_esikleme_buton = tk.Button(root, text="Webcam Görüntüsüne Otsu Eşikleme Uygula", command=webcam_otsu_esikleme)
webcam_otsu_esikleme_buton.grid(row=40, column=90)

kenarlik_ekleme_buton = tk.Button(root, text="Fotoğrafa Kenarlık Ekle", command=fotograf_kenarlik_ekleme)
kenarlik_ekleme_buton.grid(row=60, column=0)

webcam_keskinlestirme_buton = tk.Button(root, text="Webcam Görüntüsünde Keskinleştirme Yap", command=webcam_keskinlestirme)
webcam_keskinlestirme_buton.grid(row=50, column=0)


bulaniklastirma_buton = tk.Button(kontrol_frame, text="Fotoğrafa Bulanıklaştırma Uygula", command=fotograf_bulaniklastirma)
bulaniklastirma_buton.grid(row=180, column=90)
bulaniklik_derecesi_label = tk.Label(kontrol_frame, text="Bulanıklık Derecesi:")
bulaniklik_derecesi_label.grid(row=190, column=90)
bulaniklik_derecesi = tk.Scale(kontrol_frame, from_=1, to=31, orient="horizontal", length=260, tickinterval=2, command=fotograf_bulaniklastirma)
bulaniklik_derecesi.set(5)
bulaniklik_derecesi.grid(row=200, column=90)

gamma_filtreleme_buton = tk.Button(root, text="Fotoğrafa Gamma Filtreleme Uygula", command=fotograf_gamma_filtreleme)
gamma_filtreleme_buton.grid(row=40, column=0)

histogram_goster_buton = tk.Button(root, text="Fotoğrafın Histogramını Göster", command=fotograf_histogram_goster)
histogram_goster_buton.grid(row=30, column=90)

canny_kenar_tespiti_buton = tk.Button(root, text="Canny Kenar Tespiti Yap", command=canny_kenar_tespiti)
canny_kenar_tespiti_buton.grid(row=30, column=0)

Detect_Edges_btn = tk.Button(root, text='Detect Edges', command=open_image_and_detect_edges)
Detect_Edges_btn.grid(row=20, column=90)

Harris_Köşe_Tespiti_btn = tk.Button(root, text='Harris Köşe Tespiti', command=open_image_and_detect_corners)
Harris_Köşe_Tespiti_btn.grid(row=20, column=0)

Contour_Köşe_Tespiti_btn = tk.Button(root, text=' Contour Köşe Tespiti', command=open_image_and_detect_contours)
Contour_Köşe_Tespiti_btn.grid(row=10, column=90)

Watershed_algoritması_btn = tk.Button(root, text='Watershed Algoritması', command=open_image_and_apply_watershed)
Watershed_algoritması_btn.grid(row=10, column=0)




root.mainloop()
