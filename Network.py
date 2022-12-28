import numpy # библиотека для работы с матрицами, что существенно оптимизиует код, упрощаяя его для понимая
import scipy.special # библиотека для работы с функцией активации - сигмоидой  
import matplotlib.pyplot  # библиотека для визуализация данных
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
        # обновление коэффицентов связей между выходным и скрытым слоем
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
learning = 0.3
# образец сети
First = Network(inputN,hiddenN,outputN, learning)
# чтение файла содержащего ТРЕНИРОВОЧНЫЕ данные
data_file = open('C:\mnist_datase\mnist_train_100.csv', 'r') 
data_list = data_file.readlines()
# перебор всех данных
for i in  data_list:
    all_values = i.split(',') # преобразования в массив
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 # преобразование цифр из строк в числа, передача во входной сигнал
    targets = numpy.zeros(outputN) + 0.01  # создание одномерного массива длинной outputN и с минимальным значением 0.01
    targets[int(all_values[0])] = 0.99 # 'маркер' - первое значение тренировочных данных переходит в тип данных числа и принимает наибольшее значение в массиве
    First.Practice(inputs, targets) 
    pass
data_file.close()
# чтение файла содержащего ТЕСТОВЫЕ данные
test_data_file = open('C:\mnist_datase\mnist_train_100.csv', 'r') 
test_data_list = test_data_file.readlines() 
test_data_file.close()
all_values = test_data_list[0].split(',')


# проверка работы


image_array = numpy.asfarray(all_values[1:]).reshape((28,28))  # создание визуализирование цифры
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None') # наложение палитры серого на изображение
matplotlib.pyplot.show() # вывод изображения
print(First.Request((numpy.asfarray(all_values[1:])/255*0.99)+0.01)) # вывод массива с сигналами выходного слоя
results = list(First.Request((numpy.asfarray(all_values[1:])/255*0.99)+0.01))
print('Распознанная цифра: ', results.index(max(results))) # вывод результата











