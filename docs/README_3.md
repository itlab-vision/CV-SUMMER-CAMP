# Практика 3. Детектирование объектов при помощи нейронных сетей

## Цели

__Цель данной работы__ - решить задачу детектирования объектов, используя программные средства библиотеки 
OpenCV. Работа с обученными нейронными сетями.

## Задачи

__Основные задачи:__

  1. Скачать модель для детектирования объектов [mobilenet-ssd][mobilenetssd].
  1. Разработать приложение для детектирования объектов на изоражениях при помощи
  нейросети `mobilenet-ssd`.
  1. Реализовать вывод изображения на экран и отрисовку прямоугольника вокруг объектов.
  
__Дополнительные задачи:__

  1. Реализовать запись видео с прямоугольниками вокруг объектов в файл.
  1. Реализовать функции статистики (сколько объектов было в кадре, сколько времени и т.д.).
  
## Общая последовательность действий

 1. Скачать обученную сеть [mobilenet-ssd][mobilenetssd].
 1. Разработать класс `DnnDetector` (наследника класса `Detector`) для решения задачи детектирования объектов.

       - Реализовать конструктор класса `DnnDetector`, включающий инициализацию необходимых параметров.
       - Реализовать метод `Detect` класса `DnnDetector` для детектирования объектов.
          
 1. Реализовать отрисовку прямоугольников, ограничивающих детектированный объект.
 1. Выполнить [дополнительные задачи][addtasks].

## Детальная инструкция по выполнению работы

 1. в папке `<openvino_dir>`/deployment_tools/tools/model_downloader/  запустить скрипт downloader.py с параметрами --name mobilenet-ssd --output <destination_folder> 

        ```bash
        $ cd "C:\Intel\computer_vision_sdk\deployment_tools\tools\model_downloader"
        $ python downloader.py --name mobilenet-ssd --output <destination_folder>
        ```  
       В этой же папке расположен файл `list_topologies.yml`, в котором собраны параметры, перобразования входных изображений, они понаобятся для правильной конвертации вашей картинки.
        
 1. Создать рабочую ветку `practice-3` ( см. [Практика1][practice1]).
 1. Объявить класс `DnnDetector`, наследника абстрактного класса `Detector`, в файле `detector.h`.
 1. В файле `detector.сpp` реализовать конструктор класса `DnnDetector`.
 1.  В файле `detector.сpp` реализовать метод `DnnDetector::Detect`.
 

        В вашем случае выходом нейросети `mobilenet-ssd` является тензор (1x1x100x7), при вызове функции reshape(1,100) вы получите двумерную матрицу из 100 строк и 7 столбцов `[image_number, classid, score, left, bottom, right, top]`, где `image_number` - номер изображения (у нас всегда 0, так как мы подаем одно изображение); `classid` - номер класса; `score` - вероятность; `left, bottom, right, top` - координаты ограничивающих прямоугольников в диапазоне от 0 до 1. Подробный пример работы с моделью moilenet-ssd при помощи OpenCV но на языке Python можно посмотреть [по ссылке][opencv_dnn_detect]. 
  
 1. Создать копию файла `<project_source>/src/practice3.cpp` и назвать ее `<project_source>/src/practice3_YOUR_NAME.cpp`. Далее изменять код только в файле `<project_source>/src/practice3_YOUR_NAME.cpp`, но не в `<project_source>/src/practice3.cpp`.
 
 1. В файле `<project_source>/src/practice3_YOUR_NAME.cpp` реализовать детектирование лиц на изображении. Путь к файлу с изображением передается в программу с помощью ключа `--image`. Детектор загружается из файла, заданного посредством ключа `--detector`. Описание входных и выходных данных для сети можно найти на странице сети [mobilenet-ssd][mobilenetssd] и в файле `list_topologies.yml`. Полный список принимаемых параметров можно посмотреть запустив исполняемый файл с ключом `--help`. См. документацию к классу [cv::dnn::Net][opencv_dnn_net] (методы `readNet`, `setInput` и `forward`) и пространству имен [cv:dnn][opencv_dnn] (метод `blobFromImage`). Также полезным будет посмотреть примеры работы с dnn из [официальных семплов OpenCV][opencv_examples]. 
 1. Реализовать дополнительный функционал: если на вход приложению не был подан путь до изображения, использовать изображение с веб-камеры.
  
<!-- LINKS -->
[mobilenetssd]: https://github.com/chuanqi305/MobileNet-SSD
[opencv_examples]: https://docs.opencv.org/4.1.0/examples.html
[opencv_dnn]: https://docs.opencv.org/4.1.0/df/d57/namespacecv_1_1dnn.html
[opencv_dnn_net]: https://docs.opencv.org/4.1.0/db/d30/classcv_1_1dnn_1_1Net.html#details
[opencv_dnn_detect]: http://www.ebenezertechs.com/mobilenet-ssd-using-opencv-3-4-1-deep-learning-module-python/
[addtasks]: README_3.md#Задачи
[practice1]: README_1.md
