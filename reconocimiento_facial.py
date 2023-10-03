import cv2
import face_recognition as fr


ruta_base = r'D:\\SD!!!!!!!!!!!!!!!!!!!!!!!!!1\\Carrera Programacion\\06Cursos python TOTAL\\dia_14\\'

#Cargar imagenes
photos = ['foto1.jpg', 'foto2.jpg', 'foto3.jpg', 'foto4.jpg', 'foto5.jpg', 'foto6.jpg', 'foto7.jpg','foto8.jpg','foto9.jpg','foto10.jpg','foto11.jpg','foto12.jpg','foto13.jpg','foto14.jpg','foto15.jpg','foto16.jpg','foto17.jpg','foto18.jpg','foto19.jpg','Arturin.jpg']

photo_load = []
for photo in photos:
    rut_complete = ruta_base + photo
    photo = rut_complete
    load_photo = fr.load_image_file(photo)
    #una vez cargada, se necesita hacer un cambio en ambas imagenes, se necesita pasar la forma en que procesan el color, face-recognition entiene imagenes que tengan formato RGB 
    rgb_photo = cv2.cvtColor(load_photo, cv2.COLOR_BGR2RGB)
    photo_load.append(rgb_photo)


#tenemos que ayudar al sistema a reconocer en que parte de la foto hay caras
localization_face = fr.face_locations(photo_load[18])[0]#indice 0 porque se tiene que indicar que estamos enviando el primer elmeento de esa imagen
localization_face2 = fr.face_locations(photo_load[6])[0]

#codificar la imagen
codifi = fr.face_encodings(photo_load[18])[0]
codifi2 = fr.face_encodings(photo_load[6])[0]

#una vez que el programa detecte que hay una cara, debeos hacer que el usuario humano pueda ver donde se ve esa cara, que la marque
cv2.rectangle(photo_load[18], (localization_face[3], localization_face[0]), (localization_face[1], localization_face[2]), (0,255,0), 1)
cv2.rectangle(photo_load[6], (localization_face2[3], localization_face2[0]), (localization_face2[1], localization_face2[2]), (0,255,0), 1)


#Comparar caras (Debe tener parametros (objeto lista, foto a comparar) se tienen que pasar las fotos codificadas)
result = fr.compare_faces([codifi], codifi2 )
#REsultado de la comparacion
print(result)
#En ese caso mostrara un [True], ya que si se parecen, se llega a ese valor ya que el punto de comparacion o la distancia tiene un valor a 0.6



#Hay una medida que usa Face-recognition que estima la 'Distancia' que hay entre el rostro de una persona entre otra persona... si esa distancia es mayor a 0.6 no habra coincidencia, pero si es menor a 0.6 determina que si hay coincidencia. Ese valor viene por defecto pero nosotros podemos modificarlo si queremos que las coincidencias se concideran de un modo mas amplio y mas estricto

distance = fr.face_distance([codifi], codifi2)
#Tamaño de la distancia
print(distance)


#mostrar resultado

cv2.putText(photo_load[6], f'{result} {distance.round(2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0), 2)


#mostrar imagenes
for i in photo_load:
    cv2.imshow(f'Foto Control', photo_load[18])
    cv2.imshow(f'Foto Arturin', photo_load[6])

#Mantener el programa abierto
cv2.waitKey(0)
