## Описание задания

Есть 4 последовательных кадра, снятых на телефон. На этих кадрах 
нейросеть предсказала положение одного и того же знака. Нужно определить 
местоположение этого знака в мировых координатах - (latitude, longitude).

## Описание данных

### Папка images

Здесь хранятся изображения в формате jpg.

### predictions.json

В файл записана информация о предиктах знака с каждого кадра. У каждого filename
есть информация о:
- **label** - класс дорожного знака
- **x_from**, **y_from**, **width**, **height** - координаты ограничивающего бокса знака на кадре

### frames_info.json

В файл записана метаинформация для каждого кадра. В нем есть:
- **timestamp** - время записи кадра. Чтобы получить время в utc формате - нужно timestamp 
поделить на 1000
- **lat**, **lon**, **alt** - координаты места съемки. lat и lon в градусах, alt в метрах
- **azimuth** - азимут в градусах
- **speed** - скорость в км/ч
- **FocalLength** - фокусное расстояние в мм
- **FocalPlaneXResolution**, **FocalPlaneYResolution** - ширина и высота сенсора камеры в мм

### gps_info.json

Использовать этот файл необязательно, в нем записаны сырые данные gps.
Из них для каждого кадра интерполяцией получали координаты.
