# Практика 3. Детектирование объектов при помощи нейронных сетей

## Цели

__Цель данной работы__ - продолжить знакомство с программными средствами библиотеки 
OpenCV, позволяющими запускать обученные нейронные сети

__Основные задачи:__

  1. Скачать модель для распознавания головы [face-detection-adas-0001][face_detection_adas_description].
  2. Реализовать приложение, осуществляющее детектирование лиц по изображению с камеры.
  3. Реализовать вывод изображения на экран и отрисовку прямоугольника вокруг лиц.
  
__Дополнительные задачи:__

  1. Реализовать запись видео с прямоугольниками вокруг лиц в файл.
  1. Реализовать функции статистики (сколько лиц было в кадре, сколько времени и т.д.).
  
## Общая последовательность действий

 1. в папке <openvino>/deployment_tools/model_downloader/  запустить скрипт downloader.py с параметрами --name face-detection-adas-0001 --output <destination_folder> 

  ```bash
  $ cd C:\Intel\computer_vision_sdk\deployment_tools\model_downloader
  $ python downloader.py --name face-detection-adas-0001 --output <destination_folder>
  ```  
 
 2. Создать копию файла `<project_source>/src/practice3.cpp` и назвать ее `<project_source>/src/practice3_YOUR_NAME.cpp`. Далее изменять код только в файле `<project_source>/src/practice3_YOUR_NAME.cpp`, но не в `<project_source>/src/practice3.cpp`.
 
 3. В файле `<project_source>/src/practice3_YOUR_NAME.cpp` реализовать детектирование лиц на изображении. Путь к файлу с изображением передается в программу с помощью ключа `--image`. Детектор загружается из файла, заданного посредством ключа `--detector`. Описание входных и выходных данных для сети можно найти на странице сети [face-detection-adas-0001][face_detection_adas_description]. Полный список принимаемых параметров можно посмотреть запустив исполняемый файл с ключом `--help`. См. документацию к классу [cv::dnn::Net][opencv_dnn_net] (методы `readNet`, `setInput` и `forward`) и пространству имен [cv:dnn][opencv_dnn] (метод `blobFromImage`). Также полезным будет посмотреть примеры работы с dnn из [официальных семплов OpenCV][opencv_examples]. 
 4. Реализовать следующих функционал: если на вход приложению не был подан путь до изображения, использовать изображение с веб-камеры.
  
<!-- LINKS -->
[face_detection_adas_description]: https://docs.openvinotoolkit.org/latest/_face_detection_adas_0001_description_face_detection_adas_0001.html
[opencv_examples]: https://docs.opencv.org/4.1.0/examples.html
[opencv_dnn]: https://docs.opencv.org/4.1.0/df/d57/namespacecv_1_1dnn.html
[opencv_dnn_net]: https://docs.opencv.org/4.1.0/db/d30/classcv_1_1dnn_1_1Net.html#details