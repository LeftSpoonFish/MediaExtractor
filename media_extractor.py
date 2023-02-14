# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:34:55 2022

@author: IRL
"""
import os
import shutil
import glob
import fitz
import streamlit as st
import pandas as pd
import numpy as np
import magic
import cv2
import pyexiv2 as pex
import warnings

from stqdm import stqdm
from st_aggrid import AgGrid #, GridOptionsBuilder, JsCode, GridUpdateMode
from zipfile import ZipFile
from PIL import Image as Img
from mtcnn import MTCNN
from pathlib import Path

# https://blog.streamlit.io/3-steps-to-fix-app-memory-leaks/
#from memory_profiler import profile
#fp = open('safe/memory_profiler_01ME.log', 'w+')

# supress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.filterwarnings('ignore', message='This TensorFlow binary is optimized with oneAPI Deep Neural Network Library')
#warnings.filterwarnings('ignore', message='Could not load dynamic library .*libnvinfer.so.*')
#warnings.filterwarnings('ignore', message='TF-TRT Warning: Cannot dlopen some TensorRT libraries')

# suppress tensorflow logging output at launch of application
#   0 = all messages are logged (default behavior)
#   1 = INFO messages are not printed
#   2 = INFO and WARNING messages are not printed
#   3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class MediaExtractor(object):
    """
    """
#    @profile(stream=fp)
    def __init__(self, confidence=0.90, skip_frames=3, crop_margin=1.10):
        # supported file types
        self.supported_filetypes = [
            'docx', 'docm', 'dotx', 'dotm', 'xlsx', 'xlsm', 'xltx', 'xltm',
            'pptx', 'pptm', 'potm', 'potx', 'ppsx', 'ppsm', 'odt',  'ott',
            'ods',  'ots',  'odp',  'otp',  'odg',  'doc',  'dot',  'ppt',
            'pot',  'xls',  'xlt',  'pdf', 'zip', 'mp4', 'avi', 'webm', 'wmv',
            'jpeg', 'jpg', 'png', 'gif', 'bmp', 'tiff'
        ]

        # document table
        self.extract_df = pd.DataFrame(columns=['File', 'Type', 'Size', 'Count'])

        # media table
        self.media_df = pd.DataFrame(columns=['Media', 'EXIF', 'Size', 'Height', 'Width', 'Format', 'Mode', 'Hash'])        
        
        # image table
        self.image_df = pd.DataFrame(columns=['Image', 'BoxXY', 'Height', 'Width', 'Left Eye', 'Right Eye', 'Nose', 'Mouth Left', 'Mouth Right', 'IPD', 'Confidence', 'Media', 'Hash'])

        # determine file type
        self.mime = magic.Magic(mime=True)
        
        # MTCNN is used for face detection
        self.detector = MTCNN() # uses mtcnn, not dface version
        
        self.output_folder = 'output'
        self.extract_folder_name = '/extracted_images_unedited/'
        self.detection_folder_name = '/detection/'
        self.cropped_folder_name = '/cropped_faces/'
        
        # lower and upper bounds check
        if confidence < 0.25:
            self.confidence = 0.25
        elif confidence > 1.00:
            self.confidence = 1.00
        else:
            self.confidence = confidence
        
        # lower-bounds check
        if skip_frames < 1:
            self.skip_frames = 1
        else:
            self.skip_frames = skip_frames + 1
        
        # lower bounds check
        if crop_margin < 1:
            self.crop_margin = 1        
        else:
            self.crop_margin = crop_margin
        
#    @profile(stream=fp)
    def not_extract(self, file):
        """
        Unsupported file type
        """
        file_name, file_ext = os.path.splitext(file.name)
        file_type = self.mime.from_buffer(file.read())
        file_size = file.seek(0, os.SEEK_END)
        file.seek(0,0)

        metadata = {
            'File': file_name,
            'Type': file_type,
            'Size': file_size,
            'Count': 0
        }
        self.extract_df = self.extract_df.append(metadata, ignore_index=True)

        messages = {
            '.doc': 'is an older file format (Word 1997-2003). Please convert to docx or pdf from Microsoft Word.',
            '.dot': 'is an older file format (Word 1997-2003). Please convert to docx or pdf from Microsoft Word.',
            '.ppt': 'is an older file format (PowerPoint 1997-2003). Please convert to pptx or pdf from Microsoft PowerPoint.',
            '.pot': 'is an older file format (PowerPoint 1997-2003). Please convert to pptx or pdf from Microsoft PowerPoint.',
            '.xls': 'is an older file format (Excel 1997-2003). Please convert to xlsx or pdf from Microsoft Excel.',
            '.xlt': 'is an older file format (Excel 1997-2003). Please convert to xlsx or pdf from Microsoft Excel.'
        }

        message = messages.get(file_ext, 'is not a supported file type.')
        st.info(f"{file.name} {message}")
                
#    @profile(stream=fp)
    def mso_extract(self, file, location):
        """
        Extract Microsoft Office documents from 2004 to present. Current
        format is a zip file containing various subfolders.
        """
        # read enough of the data to determine the mime typ of the file
        file_type = self.mime.from_buffer(file.read())
        file.seek(0,0)
        
        # determine the size of the file read
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0,0)

        file_name = file.name
        base_name = os.path.splitext(file_name)[0]

        with ZipFile(file) as thezip:
            #st.write(thezip.infolist())
            thezip.extractall(path=location)

        # extract images
        if 'doc' in file.name:
            src = location + '/word/media/'
            root = os.path.splitext(file.name)[0]
            subfolder = os.path.splitext(file.name)[0] + self.extract_folder_name
            dest = os.path.abspath(location) + '/' + subfolder
            if os.path.exists(dest):
                shutil.rmtree(dest)
                os.makedirs(dest)
            else:
                os.makedirs(dest)               
            files = os.listdir(src)
            for f in files:
                root, ext = os.path.splitext(f)
                f_new = "{}-{}{}".format(base_name, root, ext)
                os.rename(src + f, src + f_new)
                shutil.move(src + f_new, dest)
            shutil.rmtree(location + '/word')
            metadata = {'File': file_name,
                        'Type': file_type,
                        'Size': file_size,
                        'Count': len(files)}
            self.extract_df = self.extract_df.append(metadata, ignore_index=True)

        elif 'ppt' in file.name:
            src = location + '/ppt/media/'
            root = os.path.splitext(file.name)[0]
            subfolder = os.path.splitext(file.name)[0] + self.extract_folder_name
            dest = os.path.abspath(location) + '/' + subfolder
            if os.path.exists(dest):
                shutil.rmtree(dest)
                os.makedirs(dest)
            else:
                os.makedirs(dest)
            files = os.listdir(src)
            for f in files:
                root, ext = os.path.splitext(f)
                f_new = "{}-{}{}".format(base_name, root, ext)
                os.rename(src + f, src + f_new)
                shutil.move(src + f_new, dest)
            shutil.rmtree(location + '/ppt')
            metadata = {'File': file_name,
                        'Type': file_type,
                        'Size': file_size,
                        'Count': len(files)}
            self.extract_df = self.extract_df.append(metadata, ignore_index=True)

        elif 'xl' in file.name:
            src = location + '/xl/media/'
            root = os.path.splitext(file.name)[0]
            subfolder = os.path.splitext(file.name)[0] + self.extract_folder_name
            dest = os.path.abspath(location) + '/' + subfolder
            if os.path.exists(dest):
                shutil.rmtree(dest)
                os.makedirs(dest)
            else:
                os.makedirs(dest)
            files = os.listdir(src)
            for f in files:
                root, ext = os.path.splitext(f)
                f_new = "{}-{}{}".format(base_name, root, ext)
                os.rename(src + f, src + f_new)
                shutil.move(src + f_new, dest)
            shutil.rmtree(location + '/xl')
            metadata = {'File': file_name,
                        'Type': file_type,
                        'Size': file_size,
                        'Count': len(files)}
            self.extract_df = self.extract_df.append(metadata, ignore_index=True)

        shutil.rmtree(location + '/_rels')
        shutil.rmtree(location + '/docProps')
        [os.remove(f) for f in glob.glob(location + '/*.xml')]

        return dest

#    @profile(stream=fp)
    def zip_extract(self, file, location):
        """
        https://docs.python.org/3/library/zipfile.html#module-zipfile
        """
        # get file stats
        file_name = os.path.basename(file.name)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0,0)
        
        with ZipFile(file) as thezip:

            for i in stqdm(range(len(thezip.infolist())),
                           leave=True,
                           desc='ZIP Extraction: ',
                           gui=True):

                zipinfo = thezip.filelist[i]

                file_type = os.path.splitext(zipinfo.filename)[1][1:]

                if file_type in ['doc', 'dot']:
                    st.info(f"{zipinfo.filename} is an older file format (Word 1997-2003). Please convert to docx or pdf from Microsoft Word. Open document then click save as.")
    
                elif file_type in ['ppt', 'pot']:
                    st.info(f"{zipinfo.filename} is an older file format (PowerPoint 1997-2003). Please convert to pptx or pdf from Microsoft PowerPoint. Open document then click save as.")
    
                elif file_type in ['xls', 'xlt']:
                    st.info(f"{zipinfo.filename} is an older file format (Excel 1997-2003). Please convert to xlsx or pdf from Microsoft Excel. Open document then click save as.")

                elif file_type in ['pdf']:
                    with thezip.open(zipinfo) as thefile:
                        imgpath = self.pdf_extract(thefile, self.output_folder)
                        self.__get_media(imgpath)
                        self.__get_images(imgpath)

                elif file_type in ['docx', 'docm', 'dotm', 'dotx', 'xlsx',
                                   'xlsb', 'xlsm', 'xltm', 'xltx', 'potx',
                                   'ppsm', 'ppsx', 'pptm', 'pptx', 'potm']:
                    with thezip.open(zipinfo) as thefile:
                        imgpath = self.mso_extract(thefile, self.output_folder)
                        self.__get_media(imgpath)
                        self.__get_images(imgpath)

                elif file_type in ['mp4', 'webm', 'avi', 'wmv']:
                    with thezip.open(zipinfo) as thefile:
                        imgpath = self.vid_extract(thefile, self.output_folder)
                        self.__get_media(imgpath)
                        self.__get_images(imgpath)

                    
                elif file_type in ['jpeg', 'jpg', 'png', 'gif', 'bmp', 'tiff']:
                    with thezip.open(zipinfo) as thefile:
                        imgpath = self.img_extract(thefile, self.output_folder)
                        self.__get_media(imgpath)
                        self.__get_images(imgpath)

                else:
                    pass

        metadata = {'File': file_name, 'Type': 'application/zip', 'Size': file_size, 'Count': len(thezip.infolist())}
        self.extract_df = self.extract_df.append(metadata, ignore_index=True)

#    @profile(stream=fp)
    def pdf_extract(self, file, location):
        """
        https://pymupdf.readthedocs.io/en/latest/index.html
        """
        # determine the MIME type of the file
        file_type = self.mime.from_buffer(file.read())
        file.seek(0,0)
        
        # determine the size of the file read
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0,0)

        # read the whole file into memory
        file_data = file.read()

        # open pdf file
        pdf_file = fitz.open('pdf', file_data)

        root, ext = os.path.splitext(file.name)
        subfolder = os.path.splitext(file.name)[0] + self.extract_folder_name
        document_path = os.path.abspath(location) + '/' + subfolder
        os.makedirs(document_path, exist_ok=True)

        # image counter
        nimags = 0

        # iterating through each page in the pdf
        for current_page_index in range(pdf_file.page_count):

            #iterating through each image in every page of PDF
            for img_index, img in enumerate(pdf_file.get_page_images(current_page_index)):
                  xref = img[0]
                  image = fitz.Pixmap(pdf_file, xref)
                  
                  #if it is a is GRAY or RGB image
                  if image.n < 5:        
                      image.save("{}/{}-image{}.png".format(document_path, root, nimags))

                  #if it is CMYK: convert to RGB first
                  else:                
                      new_image = fitz.Pixmap(fitz.csRGB, image)
                      new_image.writePNG("{}/{}-image{}-{}.png".format(document_path, root, nimags, img_index))
                      
                  nimags = nimags + 1

        metadata = {'File': file.name, 'Type': file_type, 'Size': file_size, 'Count': nimags}
        self.extract_df = self.extract_df.append(metadata, ignore_index=True)

        return document_path

#    @profile(stream=fp)
    def vid_extract(self, file, location):
        """
        """
        # create extraction folder
        subfolder = os.path.splitext(file.name)[0] + self.extract_folder_name
        video_path = os.path.abspath(location) + '/' + subfolder
        if os.path.exists(video_path):
            shutil.rmtree(video_path)
            os.makedirs(video_path)
        else:
            os.makedirs(video_path)

        # copy buffer to output folder
        video_file = os.path.abspath(self.output_folder) + '/' + file.name        
        with open(video_file, "wb") as f:
            f.write(file.read())

        # get file stats
        file_name = os.path.basename(video_file)
        file_type = self.mime.from_file(video_file)
        file_size = os.path.getsize(video_file)

        # initialize video frame capture        
        vidcap = cv2.VideoCapture(video_file)

        # get total number of frames
        max_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        # extract frames from video as png
        for i in stqdm(range(max_frames),
                       #st_container=st.sidebar,
                       leave=True,
                       desc='Media Extraction: ',
                       gui=True):

            # get frame
            success, image = vidcap.read()

            # break from loop is frame extraction fails
            if not success:
                break

            # write image to output path
            if i % self.skip_frames == 0:
                cv2.imwrite(video_path + os.path.splitext(file_name)[0] + f"_image{i+1}" + ".png", image)      

        # write file and media stats to dataframe
        metadata = {'File': file_name, 'Type': file_type, 'Size': file_size, 'Count': max_frames}
        self.extract_df = self.extract_df.append(metadata, ignore_index=True)

        # release video capture
        vidcap.release()
        
        # remove temporary video
        os.remove(video_file)

        return video_path
    
#    @profile(stream=fp)
    def img_extract(self, file, location):
        """
        """
        file_name, ext = os.path.splitext(os.path.basename(file.name))
        file_type = self.mime.from_buffer(file.read())
        file.seek(0,0)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0,0)

        subfolder = os.path.splitext(file_name)[0] + self.extract_folder_name
        imgpath = os.path.abspath(location) + '/' + subfolder
        if os.path.exists(imgpath):
            shutil.rmtree(imgpath)
            os.makedirs(imgpath)
        else:
            os.makedirs(imgpath) 

        metadata = {
            'File': file_name,
            'Type': file_type,
            'Size': file_size,
            'Count': 1
        }
        self.extract_df = self.extract_df.append(metadata, ignore_index=True)

        with open(imgpath + file_name + '-image1' + ext, "wb") as f:
            f.write(file.read())
            
        return imgpath

#    @profile(stream=fp)
    def crop_face(self, img, box, margin=1):
        """
        Crops facial images based on bounding box. A margin greater than one increases
        the size of the bounding box; less than one decreas the bounding box; and
        equal to one would be the bounding box size.
        """
        x1, y1, x2, y2 = box
        size = int(max(x2-x1, y2-y1) * margin)
        center_x, center_y = (x1 + x2)//2, (y1 + y2)//2
        x1, x2 = center_x-size//2, center_x+size//2
        #y1, y2 = center_y-size//2, center_y+size//2
        y1, y2 = center_y-size//1.4, center_y+size//1.4
        face = Img.fromarray(img).crop([x1, y1, x2, y2])
        return np.asarray(face)

    def face_align(self, image, keypoints):
        """
        """
        dY = keypoints['right_eye'][1] - keypoints['left_eye'][1]
        dX = keypoints['right_eye'][0] - keypoints['left_eye'][0]
        angle = np.degrees(np.arctan2(dY, dX)) #- 180

        # get the center of the image
        image_center = np.array((image.shape[1] // 2, image.shape[0] // 2), dtype=np.float32)

        # get the rotation matrix for rotating the face with no scaling
        M = cv2.getRotationMatrix2D(image_center, angle, scale=1)

        # get image dimensions
        width = image.shape[1]
        height = image.shape[0]

        image = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_CUBIC)
        
        return image

#    @profile(stream=fp)
    def __get_media(self, output_folder):
        """
        """
        try:
            files = os.listdir(output_folder)

            for f in files:
                imgfile = output_folder + '/' + f
                try:
                    im = Img.open(imgfile)
                except Exception as e:
                    st.error(e)
                    break

                # check for EXIV data
                try:
                    pimg = pex.Image(imgfile)
                    data = pimg.read_exif()
                    pimg.close()

                    if data:
                        exif_data = "yes"
                    else:
                        exif_data = "no"

                except Exception as e:
                    st.error(e)
                    break

                cropped_hash = cv2.img_hash.averageHash(cv2.imread(imgfile))[0]
                cropped_hash = ''.join(hex(i)[2:] for i in cropped_hash)

                metadata = {'Media': f,
                            'EXIF': exif_data,
                            'Size': os.path.getsize(output_folder + '/' + f),
                            'Height': im.height,
                            'Width': im.width,
                            'Format': im.format,
                            'Mode': im.mode,
                            'Hash': cropped_hash}
                self.media_df = self.media_df.append(metadata, ignore_index=True)
        except:
            pass
        
#    @profile(stream=fp)
    def __get_images(self, output_folder):
        """
        The detector returns a list of JSON objects. Each JSON object contains
        three main keys:
        - 'box' is formatted as [x, y, width, height]
        - 'confidence' is the probability for a bounding box to be matching a face
        - 'keypoints' are formatted into a JSON object with the keys:
            * 'left_eye',
            * 'right_eye',
            * 'nose',
            * 'mouth_left',
            * 'mouth_right'
          Each keypoint is identified by a pixel position (x, y).
        """
        media_files = glob.glob(output_folder + '*.*')

        face_count = 0        
        max_files = len(media_files)

        # set image path
        output_folder = output_folder.split(self.extract_folder_name)[0]
        image_path = output_folder + self.cropped_folder_name

        if os.path.exists(image_path):
            shutil.rmtree(image_path)
            os.mkdir(image_path)
        else:
            os.mkdir(image_path)

        for i in stqdm(range(max_files),
                        leave=True,
                        desc='Face Detection: ',
                        gui=True):

            f = media_files[i]
            
            try:
                image = cv2.imread(f)
                detection_image = image.copy()
                bgr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                detections = self.detector.detect_faces(bgr)
                
                # filtering detections with confidence greater than confidence threshold
                for idx, det in enumerate(detections):
                    if det['confidence'] >= self.confidence:
                        face_count = face_count + 1
                        x, y, width, height = det['box']
                        keypoints = det['keypoints']

                        # calculate ipd by taking the delta between the eyes (pixel distance)
                        ipd = keypoints['right_eye'][0] - keypoints['left_eye'][0]
                        
                        # draw bounding box for face; and points for eyes, nose and mouth
                        cv2.rectangle(detection_image, (x,y), (x+width,y+height), (0,155,255), 1)

                        # crop the image by increasing the detection bounding box (i.e. margin)
                        #  a margin of 1 is the detection bounding box
                        box = x, y, x+width, y+height
                        cropped_image = self.crop_face(image, box, margin=self.crop_margin)

                        # face alignment
                        cropped_image = self.face_align(cropped_image, keypoints)

                        # export image to disk as a PNG file
                        cropped_image_name = image_path + os.path.splitext(os.path.basename(f))[0] + '-' + f'face{idx+1}' + '.png'

                        cv2.imwrite(cropped_image_name, cropped_image)

                        # convert cropped image into an image hash using cv2
                        cropped_hash = cv2.img_hash.averageHash(cropped_image)[0]
                        cropped_hash = ''.join(hex(i)[2:] for i in cropped_hash)

                        metadata = {
                            'Image': os.path.basename(cropped_image_name),
                            'BoxXY': (x, y),
                            'Height': height,
                            'Width': width,
                            'Left Eye': keypoints['left_eye'],
                            'Right Eye': keypoints['right_eye'],
                            'Nose': keypoints['nose'],
                            'Mouth Left': keypoints['mouth_left'],
                            'Mouth Right': keypoints['mouth_right'],
                            'IPD': ipd,
                            'Confidence': det['confidence'],
                            'Media': os.path.basename(f),
                            'Hash': cropped_hash
                        }
                        
                        self.image_df = self.image_df.append(metadata, ignore_index=True)

            except Exception as e:
                st.error(e)

#    @profile(stream=fp)
    def run(self):
        """
        """
        # set streamlit page defaults
        st.set_page_config(
            layout = 'wide', # centered, wide, dashboard
            initial_sidebar_state = 'auto', # auto, expanded, collapsed
            page_title = 'Media Extractor',
            page_icon = ':eyes:' # https://emojipedia.org/shortcodes/
        )
            
        with st.form("my-form", clear_on_submit=False):
            # set title and format
            st.markdown(""" <style> .font {font-size:60px; font-family: 'Sans-serif'; text-align:center; color: blue;} </style> """, unsafe_allow_html=True)
            st.markdown('<p class="font">Media Extractor</p>', unsafe_allow_html=True)
            st.subheader('Media Input')
            self.uploaded_files = st.file_uploader("Choose a media file (image, video, or document)", type=self.supported_filetypes, accept_multiple_files=True)
            self.submitted = st.form_submit_button("PROCESS", type='primary')
                        
            if self.submitted and self.uploaded_files != []:
                max_files = len(self.uploaded_files)

                for i in stqdm(range(max_files),
                                leave=True,
                                desc='Media Extraction: ',
                                gui=True):

                    uploaded_file = self.uploaded_files[i] 

                    # split filename to get extension and remove the '.'
                    file_type = os.path.splitext(uploaded_file.name)[1][1:]

                    if file_type in ['doc', 'dot']:
                        self.not_extract(uploaded_file)

                    elif file_type in ['ppt', 'pot']:
                        self.not_extract(uploaded_file)
                        
                    elif file_type in ['xls', 'xlt']:
                        self.not_extract(uploaded_file)

                    elif file_type in ['pdf']:
                        imgpath = self.pdf_extract(uploaded_file, self.output_folder)
                        self.__get_media(imgpath)
                        self.__get_images(imgpath)

                    elif file_type in ['zip']:
                        self.zip_extract(uploaded_file, self.output_folder)
                        
                    elif file_type in ['docx', 'docm', 'dotm', 'dotx', 'xlsx',
                                       'xlsb', 'xlsm', 'xltm', 'xltx', 'potx',
                                       'ppsm', 'ppsx', 'pptm', 'pptx', 'potm']:
                        imgpath = self.mso_extract(uploaded_file, self.output_folder)
                        self.__get_media(imgpath)
                        self.__get_images(imgpath)

                    elif file_type in ['mp4', 'avi', 'webm', 'wmv']:                
                        imgpath = self.vid_extract(uploaded_file, self.output_folder)
                        self.__get_media(imgpath)
                        self.__get_images(imgpath)
                        
                    elif file_type in file_type in ['jpeg', 'jpg', 'png', 'gif', 'bmp', 'tiff']:
                        imgpath = self.img_extract(uploaded_file, self.output_folder)
                        self.__get_media(imgpath)
                        self.__get_images(imgpath)

                    else:
                        self.not_extract(uploaded_file)
                        
                st.success('Process completed')
                        
            else:
                st.info('Please select files to be processed.')

            # create metadata table
            if not self.extract_df.empty:
                st.subheader("Documents")
                AgGrid(self.extract_df, fit_columns_on_grid_load=True)
                st.info(f"* Total of {max_files} files processed, {self.extract_df['Count'].sum()} media files extracted")

            if not self.media_df.empty:
                st.subheader("Media")
                AgGrid(self.media_df, fit_columns_on_grid_load=True)

            if not self.image_df.empty:
                st.subheader("Images")
                AgGrid(self.image_df, fit_columns_on_grid_load=True)
                st.info(f"* Found a total of {len(self.image_df)} face(s) in media files")

if __name__ == '__main__':
    from streamlit_profiler import Profiler
    with Profiler():
        mx = MediaExtractor()    
        mx.run()

