import cv2
import matplotlib.pyplot as plt

def local_histogram_equalization(image, window_height, window_width):
    # Obtenemos las dimensiones de la imagen
    rows, cols = image.shape

    # Calculamos la mitad de la anchura y altura de la imagen. Esto es para crear los bordes luego a la imagen original
    # y asi no tener problemas cuando la ventana pase por los píxeles pegados a estos.
    half_height = (window_height - 1) // 2 if window_height % 2 != 0 else window_height // 2
    half_width = (window_width - 1) // 2 if window_width % 2 != 0 else window_width // 2

    # Aplicamos un filtro de mediana para reducir el ruido
    image = cv2.medianBlur(image, 3)

    # Agregamos bordes a la imagen para evitar problemas en los propios bordes de la imagen
    padded_image = cv2.copyMakeBorder(image, half_height, half_height, half_width, half_width, cv2.BORDER_REPLICATE)

    for i in range(rows):
        for j in range(cols):
            # Obtenemos la ventana local alrededor del píxel actual
            window = padded_image[i:i+window_height, j:j+window_width]

            # Calculamos el histograma de la ventana
            hist = cv2.calcHist([window], [0], None, [256], [0, 256])

            # Realizmos la ecualización del histograma local
            cdf = hist.cumsum()
            cdf_normalized = cdf / cdf[-1] # Normalizamos la función acumulativa del histograma
            transformed_pixel_value = cdf_normalized[window[half_height, half_width]] * 255

            # Asignamos el valor transformado al píxel en la imagen
            image[i, j] = transformed_pixel_value

    return image

# Cargamos la imagen de entrada
input_image = cv2.imread('Imagenes\Problema_1\Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)

# Tamaño de la ventana para el procesamiento local
window_height = 5
window_width = 5

# Aplicamos la ecualización local del histograma sobre la imagen
input_image = local_histogram_equalization(input_image, window_height, window_width)

# Mostramos la imagen resultante
plt.imshow(input_image, cmap='gray')
plt.title('Imagen Ecualizada Local')
plt.axis('off')  # Ocultamos los ejes
plt.show()