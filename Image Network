import numpy # библиотека для работы с матрицами, что существенно оптимизиует код, упрощаяя его для понимая
import scipy.special # библиотека для работы с функцией активации - сигмоидой  
import matplotlib.pyplot  # библиотека для визуализация данных
import imageio.v2 # библиотека для чтения фотографий и работы с ней
import glob # библиотека для изменения фотографии
from PIL import Image 
class Network():
    # функция инициализации сети
    def  __init__(self, inputN, hiddenN, outputN, learning):
        # количество узлов во входном слое, скрытом слое и выходном слое
        self.iN = inputN
        self.hN = hiddenN
        self.oN = outputN
        # коэффицент обучения
        self.lr = learning
        # матрица весовых коэффицентов 
        self.Wih = numpy.random.normal(0.0, pow(self.iN, -0.5), (self.hN, self.iN))
        self.Who = numpy.random.normal(0.0, pow(self.hN, -0.5), (self.oN, self.hN))
        # функция сигмоиды
        self.activation_F = lambda x: scipy.special.expit(x) 

        pass
    # функция  тренировки сети
    def Practice(self, inputs_list, targets_list):
        # преобразовать список тренировочных значений в  двухмерный массив 
        targets = numpy.array(targets_list, ndmin=2).T
        # создание матрицы входных сигналов
        inputs = numpy.array(inputs_list, ndmin=2).T 
        # расчет сигнала на скрытом слое
        hidden_inputs = numpy.dot(self.Wih, inputs) # сигналы входящие на скрытый слой
        hidden_outputs = self.activation_F(hidden_inputs) # сигналы исходящие из скрытого слоя
        # расчет сигнала на выходном слое
        final_inputs = numpy.dot(self.Who, hidden_outputs) # сигналы входящие на выходной слой
        final_outputs = self.activation_F(final_inputs) # сигналы исходящие из выходного слоя
        # ошибки 
        output_errors = targets - final_outputs # суть ошибки
        hidden_errors = numpy.dot(self.Who.T, output_errors) # матрица оишбок на скрытом слое
        # обновление весовых коэффицентов 
        self.Who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs)) # между скрытым и выходным
        self.Wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs)) # между входным и скрытым

        pass
    # функция  обратботки сигналов сети
    def Request(self, inputs_list):
        # создание матрицы входных сигналов
        inputs = numpy.array(inputs_list, ndmin=2).T 
        # расчет сигнала на скрытом слое
        hidden_inputs = numpy.dot(self.Wih, inputs) # сигналы входящие на скрытый слой
        hidden_outputs = self.activation_F(hidden_inputs) # сигналы исходящие из скрытого слоя
        # расчет сигнала на выходном слое
        final_inputs = numpy.dot(self.Who, hidden_outputs) # сигналы входящие на выходной слой
        final_outputs = self.activation_F(final_inputs) # сигналы исходящие из выходного слоя
        
        return  final_outputs 


# создание сети


# сигналы
inputN = 784 # размер изображения представляет собойквадрат 28х28 пикселей
hiddenN = 200 # количество тренировчоных примеров, выбрав значения меньше входных узлов, нейросеть старается обобщить информацию
outputN= 10 # количество цифр в десятичной системе исчисления
# коэффицент обучения
learning = 0.2
# образец сети
First = Network(inputN,hiddenN,outputN, learning)
# чтение файла содержащего ТРЕНИРОВОЧНЫЕ данные
data_file = open('C:\mnist_datase\mnist_train.csv', 'r') 
data_list = data_file.readlines()
# перебор всех данных
stage = 5 # количество поколений - перебора тренировочных данных
for  age in range(stage):
    for i in  data_list:
        all_values = i.split(',') # преобразования в массив
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 # преобразование цифр из строк в числа, передача во входной сигнал
        targets = numpy.zeros(outputN) + 0.01  # создание одномерного массива длинной outputN и с минимальным значением 0.01
        targets[int(all_values[0])] = 0.99 # 'маркер' - первое значение тренировочных данных переходит в тип данных числа и принимает наибольшее значение в массиве
        First.Practice(inputs, targets)
        pass 
    pass
data_file.close()



# ТЕСТ



# преобразование фотографии для нейронной сети
for photo in glob.glob('C:\mnist_datase\picture_?.png'): # сжатие фотографии до формата 28х28
    image = Image.open(photo)
    new_image = image.resize((28,28))
    new_image.save(photo)


# тестирование


test = []
count = 0
for image_file_name in glob.glob('C:\mnist_datase\picture_?.png'): # обработка фотографии
    label = int(image_file_name[-5:-4]) # поиск маркера для фотографии
    print ("Загружаем фото файла ...",image_file_name) # вывести файл, с которым идет работа
    img_array = imageio.v2.imread(image_file_name, as_gray=True)
    img_data  = 255.0 - img_array.reshape(784) # инверсия цветовой палитры и сжатие изображения
    img_data = (img_data / 255.0 * 0.99) + 0.01 # масштабирование данных
    record = numpy.append(label,img_data)
    test.append(record)
    count += 1
    pass


# тест
number = 0
while number < count:
    matplotlib.pyplot.imshow(test[number][1:].reshape(28,28), cmap='Greys', interpolation='None')
    matplotlib.pyplot.show()
    correct_label = test[number][0] # маркер для данной записи
    inputs = test[number][1:] # перебор всех значений и отправление сигналов в сеть
    outputs = First.Request(inputs) # опрос сети
    print (outputs) # массив с выходными сигналами
    label = numpy.argmax(outputs)
    print("Ответ нейронной сети: ", label)
    if (label == correct_label):
        print ("Совпадение")
    else:
        print ("Верный ответ:", correct_label)
    number += 1
    pass
