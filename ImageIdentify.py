import face_recognition as faces
from PIL import Image, ImageDraw
import os

#create array of encodings and names
know_encode = []

know_names = []

#the grasita function
def readImages():
    obj = os.scandir('./img/known')
    for entry in obj:
        n = entry.name
        know_names.append(n[:-4])
        img = faces.load_image_file('./img/known/'+n)
        i_encode = faces.face_encodings(img)[0]
        know_encode.append(i_encode)
        print(n[:-4] + ' proccess')

readImages()

#load test image
test_image1 = faces.load_image_file('./img/groups/vladimir-putin-bashar-al-asad-y-barack-obama-1.jpg')

# find faces in test images
face_location1 = faces.face_locations(test_image1)
face_encode1 = faces.face_encodings(test_image1, face_location1)

# convert to PIL format
pil_image1 = Image.fromarray(test_image1)

#create a draw instance
draw1 = ImageDraw.Draw(pil_image1)


#loop the faces in images
for(top, right, bottom, left), face_encode1 in zip(face_location1, face_encode1):
    matches = faces.compare_faces(know_encode, face_encode1)
    name = "Unknown Person"

    if True in matches:
        first_match_index = matches.index(True)
        name = know_names[first_match_index]

    #draw box
    draw1.rectangle(((left, top), (right, bottom)), outline=(0,0,0))

    #draw label
    text_width, text_height = draw1.textsize(name)
    draw1.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0,0,0), outline=(0,0,0))
    draw1.text((left+6, bottom - text_height - 5), name, fill=(255,255,255,255))

#display image
pil_image1.show()

del draw1