import cv2
import face_recognition as fr
import os
import numpy
import datetime
 
#Data Base Employees
route_root = r'D:\\SD!!!!!!!!!!!!!!!!!!!!!!!!!1\\Carrera Programacion\\06Cursos python TOTAL\\dia_14\\Empleados\\'
 
photos_employees = []
names_employees = []
list_employees = os.listdir(route_root)
 
for name in list_employees:
    imgage_act = cv2.imread(os.path.join(route_root,name))
    photos_employees.append(imgage_act)
    names_employees.append(f'{os.path.splitext(name)[0]}')
    
 
#Convert from BGR to RGB and encoding

def encoding(photos):
    #list employees encogding
    list_encoding = []
    #list RGB
    
    for img in photos:
        #convert RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #encoding photo
        img_enc = fr.face_encodings(img_rgb)
        if img_enc:
            list_encoding.append(img_enc[0])
        
    return(list_encoding)
 
photo_encoding_def = encoding(photos_employees)
 
 #register asist
registro_file_path = os.path.join(r'D:\\SD!!!!!!!!!!!!!!!!!!!!!!!!!1\\Carrera Programacion\\06Cursos python TOTAL\\dia_14\\registro.csv')
def register(employe):
    f = open(registro_file_path, 'r+')
    lista_date = f.readlines()
    name_register = []
    for line in lista_date:
        income = line.split(',')
        name_register.append(income[0])
    if employe not in name_register:
        hour = datetime.datetime.now()
        str_now = hour.strftime('%H:%M:%S')
        f.writelines(f'\n {employe}, {str_now}')
    else:
        print('Ya te registraste')
        
#Shot photo Webcam
 
shot = cv2.VideoCapture(0)
 
#read photo webcam
succes, img = shot.read()
 
if not succes:
    print('No se pudo capturar la foto')
else:
    #recognition
    face_loca = fr.face_locations(img)
    #encoding face
    face_encodign = fr.face_encodings(img, face_loca)
    #coincidences
    
    for facecodif, faceloc in zip (face_encodign, face_loca):
        coincidence = fr.compare_faces(photo_encoding_def, facecodif)
        distance = fr.face_distance(photo_encoding_def, facecodif)
        print(distance)
        
        index_coincidence = numpy.argmin(distance)

        #show coincidences
        if distance[index_coincidence] > 0.6:
            print('No hay coincidencias') 
        else:
            
            #search name to emloyee
            name = names_employees[index_coincidence]
            
            y1, x2, y2, x1 = faceloc
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 1)
            cv2.rectangle(img, (x1,y2 - 50),(x2,y2),(0,255,0), 1)   
            cv2.putText(img, name, (x1 +6, y2-6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1)
                    
            register(name)
            #show img
            cv2.imshow('Employee', img)
            
            cv2.waitKey(0)
            
         
