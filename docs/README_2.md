# Практика 2. Классификация изображений при помощи нейронных сетей

## Цели

__Цель данной работы__ - познакомиться с программными средствами библиотеки 
OpenCV, позволяющими запускать обученные нейронные сети

## Задачи
  
__Основные задачи:__

  1. С помощью скрипта, входящего в состав OpenVINO, скачать обученную сеть squeezenet1.1.
  2. Реализовать приложение, осуществляющее классификацию изображений.
  3. Реализовать вывод приложением top-3 (top-5, top-10) классов, предложенных классификатором.  
  
__Дополнительные задачи:__

  1. Добавить возможность классификации не по целому изображению, а по его части, заданной пользователем.
  
## Общая последовательность действий
  
  1. в папке <openvino>/deployment_tools/model_downloader/  запустить скрипт downloader.py с параметрами --name squeezenet1.1 --output <destination_folder> 
  ```bash
  $ cd C:\Intel\computer_vision_sdk\deployment_tools\model_downloader
  $ python downloader.py --name squeezenet1.1 --output <destination_folder>
  ```  
 
  2. Создать копию файла `<project_source>/src/practice2.cpp` и назвать ее `<project_source>/src/practice2_YOUR_NAME.cpp`. Далее изменять код только в файле `<project_source>/src/practice2_YOUR_NAME.cpp`, но не в `<project_source>/src/practice2.cpp`.
  
  3. Убедиться, что проект успешно собирается и создается новый исполняемый файл `<project_build>/bin/practice2_YOUR_NAME.exe`.
  
  4. Прислать Pull Request с внесенными изменениями. Пометить в конце названия `(NOT READY)`. По мере готовности решений основных задач Pull Request можно будет переименовать.
  
  5. В файле `<project_source>/src/practice2_YOUR_NAME.cpp` реализовать классификацию изображения. Путь к файлу с изображением передается в программу с помощью ключа `--image`. Детектор загружается из файла, заданного посредством ключа `--detector`. Полный список принимаемых параметров можно посмотреть запустив исполняемый файл с ключом `--help`. См. документацию к классу [cv::dnn::Net][opencv_dnn_net] (методы `readNet`, `setInput` и `forward`) и пространству имен [cv:dnn][opencv_dnn] (метод `blobFromImage`). Также полезным будет посмотреть примеры работы с dnn из [официальных семплов OpenCV][opencv_examples]
  
  6. Решить задачи из списка [Дополнительные задачи][tasks].

<!-- LINKS -->
  
[practice1]: docs/README_1.md
[git-intro]: docs/README_1.md#Общие-инструкции-по-работе-с-git
[cmake-msvs]: docs/README_1.md#Сборка-проекта-с-помощью-cmake-и-microsoft-visual-studio 
[opencv_dnn]: https://docs.opencv.org/4.1.0/df/d57/namespacecv_1_1dnn.html
[opencv_dnn_net]: https://docs.opencv.org/4.1.0/db/d30/classcv_1_1dnn_1_1Net.html#details
[opencv_examples]: https://docs.opencv.org/4.1.0/examples.html
[opencv_dnn_classification_sample]: https://docs.opencv.org/4.1.0/d9/d8d/samples_2dnn_2classification_8cpp-example.html