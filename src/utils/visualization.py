import functools
import io
import os
from typing import NamedTuple, Tuple
import xml.etree.ElementTree as Xml

import cv2
import geojson
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter,ImageFont, ImageOps
from pytorch_lightning.callbacks import BasePredictionWriter
from scipy.stats import rankdata
import seaborn as sn
from torchvision.transforms import ToTensor

WSI_EXTENSIONS = [".tif", ".svs"]

class ScatterPlotter(object):
    def __init__(self) -> None:
        pass
    

class PredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        write_interval,
        raw_root_dir: str,
        processed_root_dir: str=None,
        output_dir: str=None,
    ):
        super().__init__(write_interval)
        self.illustrator = Illustrator(
            raw_root_dir=raw_root_dir,
            processed_root_dir=processed_root_dir,  
            output_dir = output_dir                         
            )

    def write_on_batch_end(
        self, 
        trainer,
        pl_module, 
        prediction, 
        batch_indices, 
        batch, 
        batch_idx, 
        dataloader_idx
    ):   
        preds = prediction['preds']
        attention = prediction['attention']
        self.illustrator.save_attention_maps(prediction=prediction, attention=attention)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        for prediction in predictions:
            preds = prediction['preds']
            attention = prediction['attention']
            self.illustrator.save_attention_maps(prediction=prediction, attention=attention)


class Illustrator(object):
    def __init__(
        self,
        raw_root_dir: str,
        processed_root_dir: str=None,
        output_dir: str=None,
        thumbnail_level: int=3,
        score_threshold: float=0.0
        ) -> None:
        self.raw_root_dir = raw_root_dir
        self.processed_root_dir = processed_root_dir
        self.output_dir = output_dir
        self.thumbnail_level = thumbnail_level
        self.score_threshold = score_threshold

    def get_filepath(self, slide_id):
        slide_id_without_staining = "_".join(slide_id.split("_")[:2])
        if "TCGA" in slide_id:
            slide_id_without_staining = slide_id.split("_")[0]
        filepath = [os.path.join(root, filename) 
                    for root, _ ,filenames in os.walk(self.raw_root_dir+"/", topdown=True, followlinks=True) 
                    for filename in filenames 
                    for ext in WSI_EXTENSIONS 
                    if filename.endswith(ext) and slide_id in filename
                    ]
        if not filepath:
            filepath = [os.path.join(root, filename) 
                        for root, _ ,filenames in os.walk(self.raw_root_dir+"/", topdown=True, followlinks=True) 
                        for filename in filenames 
                        for ext in WSI_EXTENSIONS 
                        if filename.endswith(ext) and slide_id_without_staining in filename
                        ]
        assert len(filepath)==1, f"Found more than one match, {filepath}"
        return filepath[0]
    
    def get_slide(self, slide_id):
        filepath = self.get_filepath(slide_id)
        return UKTSlide(filepath=filepath, thumbnail_level=self.thumbnail_level, mask_dir=os.path.join(self.processed_root_dir, "masks"))

    # def create_attention_maps(self, attention):
    #     keys = attention['keys']
    #     scores_raw = attention['scores']
    #     n_classes = scores_raw.shape[0]
    #     slide_id = keys[0][0].split("_level")[0]
    #     slide = self.get_slide(slide_id=slide_id)
    #     patch_size = 256 * 2 ** slide.level_offset
        
    #     shape = slide.thumb.shape[:2]
    #     scale_factor = 2 ** slide.thumbnail_level

    #     x_coords = np.array([int(key[0].split("_y")[0].split("x")[1]) for key in keys]) - patch_size/2
    #     y_coords = np.array([int(key[0].split("y")[1]) for key in keys]) - patch_size/2
    #     rel_coords = np.array([x_coords, y_coords]) / scale_factor
    #     rel_patch_size = int(patch_size / scale_factor)

    #     # Global normalization based on the ranking of all classes
    #     scores_abs = scores_raw.flatten()
    #     scores_abs = rankdata(scores_abs.cpu().numpy(), 'average')/len(scores_abs) * 100
    #     scores_abs = scores_abs.reshape((n_classes, -1))
    #     # ALTERNATIVE class specific ranking:
    #     scores_rel = np.array([rankdata(scores_raw[n,:].cpu().numpy(), 'average')/len(scores_raw[n,:]) * 100 for n in range(n_classes)])
    #     scores = np.stack([scores_abs, scores_rel])
    #     grid = []
    #     for i in range(len(scores)):
    #         attention_map = np.zeros((n_classes, *shape), dtype=float)
    #         for n in range(n_classes):     
    #             for score, (x, y) in zip(scores[i,n,:], rel_coords.transpose()):
    #                 x = int(x)
    #                 y = int(y)
    #                 attention_map[n, y:y+rel_patch_size, x:x+rel_patch_size] += score/100
    #             attention_maps = []
    #             for n in range(n_classes):
    #                 a_map = Image.fromarray(np.uint8(cm.jet(attention_map[n])*255)).convert("RGB")
    #                 attention_maps.append(a_map)
    #             attention_maps = np.hstack(attention_maps)
    #             # Create tissue contours and store in overlay
    #             tissue_mask = slide.tissue_mask
    #             tissue_mask = Image.fromarray((tissue_mask*255).astype(np.uint8)).convert("RGB")
    #             tissue_mask_contour = tissue_mask.filter(ImageFilter.FIND_EDGES)
    #             tissue_mask_contour = np.array(tissue_mask_contour)
    #             overlay = np.array(slide.thumb)*(tissue_mask_contour==0)*1
    #             overlay += np.array([0,1,0]).astype('uint8')*tissue_mask_contour
    #             if slide.is_annotated:
    #                 # Create cancer contours and store in overlay
    #                 cancer_mask = slide.cancer_mask
    #                 cancer_mask = Image.fromarray((cancer_mask*255).astype(np.uint8)).convert("RGB")
    #                 cancer_mask_contour = cancer_mask.filter(ImageFilter.FIND_EDGES)
    #                 cancer_mask_contour = np.array(cancer_mask_contour)
    #                 overlay = overlay*(cancer_mask_contour==0)*1
    #                 overlay += np.array([1,0,0]).astype('uint8')*cancer_mask_contour
    #         grid.append(np.hstack([overlay, attention_maps]))
    #     grid = np.vstack(grid)
    #     return slide_id, grid

    def create_attention_maps(self, attention, cmap="jet"):
        keys = attention['keys']
        scores_raw = attention['scores']
        n_classes = scores_raw.shape[0]
        slide_id = keys[0][0].split("_level")[0]
        slide = self.get_slide(slide_id=slide_id)
        overlay = self.get_overlay(slide=slide)
        
        # Get relative coordinates
        if "TCGA" in slide.name:
            patch_size = 256 * slide.level_downsamples[slide.level_offset]
            scale_factor = slide.level_downsamples[slide.thumbnail_level]
        else:
            patch_size = 256 * 2 ** slide.level_offset
            scale_factor = 2 ** slide.thumbnail_level

        x_coords = np.array([int(key[0].split("_y")[0].split("x")[1]) for key in keys]) - patch_size/2
        y_coords = np.array([int(key[0].split("y")[1]) for key in keys]) - patch_size/2
        rel_coords = np.array([x_coords, y_coords]) / scale_factor
        rel_patch_size = int(patch_size / scale_factor)

        # Global normalization based on the ranking of all classes
        scores_abs = scores_raw.flatten()
        scores_abs = rankdata(scores_abs.cpu().numpy(), 'average')/len(scores_abs) * 100
        scores_abs = scores_abs.reshape((n_classes, -1))
        # ALTERNATIVE class specific ranking:
        scores_rel = np.array([rankdata(scores_raw[n,:].cpu().numpy(), 'average')/len(scores_raw[n,:]) * 100 for n in range(n_classes)])
        scores = np.stack([scores_abs, scores_rel])
        # scores = np.array([[(scores_raw[n,:].cpu().numpy()-np.min(scores_raw[n,:].cpu().numpy()))/(np.max(scores_raw[n,:].cpu().numpy())-np.min(scores_raw[n,:].cpu().numpy())) for n in range(n_classes)]])
        
        attention_maps = []
        shape = slide.thumb.shape[:2]
        for i in range(len(scores)):
            attention_map = np.zeros((n_classes, *shape), dtype=float)
            for n in range(n_classes):     
                for score, (x, y) in zip(scores[i,n,:], rel_coords.transpose()):
                    x = int(x)
                    y = int(y)
                    attention_map[n, y:y+rel_patch_size, x:x+rel_patch_size] += score/100
                    # attention_map[n, y:y+rel_patch_size, x:x+rel_patch_size] += score
            attention_maps.append(attention_map)
        return slide_id, overlay, attention_maps

    def get_overlay(self, slide):
                # Create tissue contours and store in overlay
        thumb = np.array(slide.thumb)
        # tissue_mask = slide.tissue_mask
        # tissue_mask = Image.fromarray((tissue_mask*255).astype(np.uint8)).convert("RGB").resize(thumb.shape[::-1][1:])
        # tissue_mask_contour = tissue_mask.filter(ImageFilter.FIND_EDGES)
        # tissue_mask_contour = np.array(tissue_mask_contour)
        # overlay = thumb*(tissue_mask_contour==0)*1
        # overlay += np.array([0,1,0]).astype('uint8')*tissue_mask_contour
        if slide.is_annotated:
            # Create cancer contours and store in overlay
            cancer_mask = slide.cancer_mask
            cancer_mask = Image.fromarray((cancer_mask*255).astype(np.uint8)).convert("RGB")
            cancer_mask_contour = cancer_mask.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(7))
            cancer_mask_contour = np.array(cancer_mask_contour)
            overlay = thumb*(cancer_mask_contour==0)*1
            overlay += np.array([0,128,0]).astype('uint8')*cancer_mask_contour
            return overlay
        else:
            return thumb

    # def save_attention_maps(self, attention, prediction, target=None):
    #     os.makedirs(self.output_dir, exist_ok=True)
    #     slide_id, attention_maps = self.create_attention_maps(attention=attention)
    #     attention_maps = Image.fromarray(attention_maps).convert("RGB")
    #     attention_maps = ImageOps.expand(attention_maps, border=140, fill=(255,255,255))
    #     draw = ImageDraw.Draw(attention_maps)
    #     font = ImageFont.truetype("arial.ttf", 50)
    #     if target is not None:
    #         line = f"{slide_id} \nTarget: {int(target)} Prediciton: {int(prediction)}"
    #         w, _ = draw.textsize(line, font=font)
    #         position = (int(attention_maps.width/2)-int(w/2), 20)
    #         draw.text(position, line, (0,0,0), font=font, spacing=6, align ="center")
    #         attention_maps_path = os.path.join(self.output_dir,f"{slide_id}_target{int(target)}_pred{int(prediction)}.png")
    #     else:
    #         line = f"{slide_id} \n Prediciton: {int(prediction)}"
    #         w, _ = draw.textsize(line, font=font)
    #         position = (int(attention_maps.width/2)-int(w/2), 20)
    #         draw.text(position,line,(0,0,0),font=font, spacing=6, align ="center")
    #         attention_maps_path = os.path.join(self.output_dir,f"{slide_id}_pred{int(prediction)}.png")    
    #     attention_maps.save(attention_maps_path)

    def save_attention_maps(self, attention, prediction, target=None):
        os.makedirs(self.output_dir, exist_ok=True)
        slide_id, annotated_thumb, attention_maps = self.create_attention_maps(attention=attention)
        annotated_thumb = Image.fromarray(annotated_thumb).convert("RGB")
        annotated_thumb_path = os.path.join(self.output_dir,f"{slide_id}.png")
        annotated_thumb.save(annotated_thumb_path)
        if target is not None:
            for n, attention_map in enumerate(attention_maps[0]):
                attention_map[attention_map<self.score_threshold]=0
                cmap = cm.get_cmap('jet')
                attention_map = Image.fromarray(np.uint8(cmap(attention_map)*255)).convert("RGB")
                attention_map.save(os.path.join(self.output_dir,f"{slide_id}_am_class{n}_target{int(target)}_pred{int(prediction)}_{self.score_threshold}_{self.thumbnail_level}.png"))
                overlay = Image.blend(annotated_thumb, attention_map, 0.2)
                overlay.save(os.path.join(self.output_dir,f"{slide_id}_class{n}_target{int(target)}_pred{int(prediction)}_{self.score_threshold}_{self.thumbnail_level}.png"))
        # attention_maps = Image.fromarray(attention_maps).convert("RGB")
        # attention_maps = ImageOps.expand(attention_maps, border=140, fill=(255,255,255))
        # draw = ImageDraw.Draw(attention_maps)
        # font = ImageFont.truetype("arial.ttf", 50)
        # if target is not None:
        #     line = f"{slide_id} \nTarget: {int(target)} Prediciton: {int(prediction)}"
        #     w, _ = draw.textsize(line, font=font)
        #     position = (int(attention_maps.width/2)-int(w/2), 20)
        #     draw.text(position, line, (0,0,0), font=font, spacing=6, align ="center")
        #     attention_maps_path = os.path.join(self.output_dir,f"{slide_id}_target{int(target)}_pred{int(prediction)}.png")
        # else:
        #     line = f"{slide_id} \n Prediciton: {int(prediction)}"
        #     w, _ = draw.textsize(line, font=font)
        #     position = (int(attention_maps.width/2)-int(w/2), 20)
        #     draw.text(position,line,(0,0,0),font=font, spacing=6, align ="center")
        #     attention_maps_path = os.path.join(self.output_dir,f"{slide_id}_pred{int(prediction)}.png")    
        # attention_maps.save(attention_maps_path)

class Annotation(NamedTuple):
    name: str
    polygon: Tuple


class Point(NamedTuple):
    x: int
    y: int


class UKTSlide(openslide.OpenSlide):
    """
    Custom whole slide image class, which inherits from OpenSlide
    """
    def __init__(
            self, 
            filepath,
            mask_dir: None,
            thumbnail_level: int=6,
        ):
        """_summary_

        Args:
            filepath (str): filepath to whole slide image
            thumbnail_level (int): downsamling level to extract thumbnail. Defaults to 6.
            mask_dir (str): directory with corresponding (tissue) masks
        """
        super().__init__(filepath)
        self.filepath = filepath
        self.name = os.path.basename(self.filepath).split('.')[0]
        if len(self.name.split('_'))>2:
            self.staining = self.name.split('_')[2]
        else:
            self.staining ="HE"
        self.mask_dir = mask_dir
        self.mpp = self.properties["openslide.mpp-x"]
        # if "tiff.ImageDescription" in self.properties:
        #     self.description = self.properties["tiff.ImageDescription"]
        #     # Assure the same pixel resolution for all whole slide images 
        #     if "mag=40" in self.description:
        #         self.level_offset=1
        # else:
        #     self.level_offset=0
        if "tiff.ImageDescription" in self.properties:
            self.description = self.properties["tiff.ImageDescription"]
        else:
            self.description = None
        # Assure the same pixel resolution for all whole slide images 
        if self.description and "mag=20" in self.description:
            self.level_offset=0
        else:
            self.level_offset=1
        self.thumbnail_level = min((thumbnail_level+self.level_offset), len(self.level_dimensions)-1)
        self.width = self.dimensions[0]
        self.height = self.dimensions[1]
        self._thumb = None
        self._tissue_mask = None
        self._cancer_mask = None
        self._annotations = None
        self.annotation_ext = ".xml"
        # If HE slide check whether annotation file exists
        self.is_annotated = (os.path.isfile(self.filepath.split('.')[0] + self.annotation_ext) and self.staining=="HE")
        
        
    @property
    def thumb(self):
        """Returns a numpy array containing a thumbnail of the slide"""
        if self._thumb is None:
            self._thumb = self.get_thumb()
        return self._thumb
        # return self.get_thumb()
    
    @property
    def tissue_mask(self):
        """Returns a binary numpy array containing the corresponding tissue regions of the slide"""
        if self._tissue_mask is None:
            tissue_mask_path = os.path.join(self.mask_dir, self.name+'_tissue.png')
            assert os.path.exists(tissue_mask_path), f"Tissue mask not found {tissue_mask_path}"
            self._tissue_mask = self.get_tissue_mask(tissue_mask_path)
        return self._tissue_mask

    @property
    def annotations(self) -> Tuple[Annotation]:
        """ Transforms slide corresponding annotations stored in xml- or geojson-files into a tuple of Annotations"""
        if self._annotations is None:
            if self.is_annotated:
                annotation_path = self.filepath.split('.')[-2] + ".xml"
                annotations = []
                if self.annotation_ext == ".xml":
                    tree = Xml.parse(annotation_path)
                    root = tree.getroot()
                    for annotation in root.iter('Annotation'):
                        # all annotation points in sorted by the `Order` attribute
                        annotations.append(Annotation(annotation.attrib['Name'].replace(' ', ''),
                                                    tuple([Point(float(c.attrib['X']), float(c.attrib['Y']))
                                                            for c in sorted(annotation.iter('Coordinate'),
                                                                            key=lambda x: int(x.attrib['Order']))])))
                if self.annotation_ext == ".geojson":
                    annotation_path = self.filepath.split('.')[0] + self.annotation_ext
                    with open(annotation_path) as f:
                        data = geojson.load(f)
                    for annotation in data['features']:
                        annotations.append(Annotation(annotation['properties']['classification']['name'], 
                                           tuple([Point(float(c[0]), float(c[1])) for c in annotation['geometry']['coordinates'][0]])))
                self._annotations = tuple(annotations)
            else:
                self._annotations = ()
        return self._annotations
    
    @property
    def cancer_mask(self):
        """Returns a binary numpy array containing the corresponding cancer regions of the slide"""
        if self._cancer_mask is None:
            self._cancer_mask = self.get_cancer_mask()
            # cancer_map = Image.fromarray((self.cancer_map* 255).astype(np.uint8))
            # cancer_map.save(os.path.join(self.path.split('training/')[0], 'training/cancer_maps', os.path.basename(self.path).split('.')[0]+'_cancer.png'))
        return self._cancer_mask

    def get_thumb(self, grey=False):
        """Returns a numpy array containing a thumbnail of the image, either greyscale or RGB

        Args:
            grey (bool, optional): Flag to convert the RGB thumbnail to greyscale. Defaults to False.         
        """
        if grey:
            return np.asarray(self.get_thumbnail(self.level_dimensions[self.thumbnail_level]).convert('L'))
        return np.asarray(self.get_thumbnail(self.level_dimensions[self.thumbnail_level]).convert('RGB'))
    
    def get_tissue_mask(self, tissue_mask_path):
        with Image.open(tissue_mask_path).convert('L') as tissue_image:
                tissue_mask = np.asarray(tissue_image)
                # Make sure to normalize the mask!
                if tissue_mask.max() != 1.0:
                    tissue_mask = tissue_mask / tissue_mask.max()
                # return np.asarray(tissue_image)
                tissue_image.close()
        return tissue_mask.astype(int)
    
    def get_cancer_mask(self):
        left_upper=(0, 0)
        shape = self.thumb.shape[:2]
        if len(shape) > 2:
            shape = shape[:2]
        cancer_mask = np.zeros(shape, dtype=np.uint8)
        # First get cancer mask
        if self.is_annotated:
            scale_factor = 2 ** self.thumbnail_level
            for _, annotation in enumerate(self.annotations):
                polygon = np.asarray(list(zip(*annotation.polygon))).transpose()
                rel_polygon = (polygon - left_upper) / scale_factor
                rel_polygon.astype('int32')
                cv2.fillPoly(cancer_mask, pts=[rel_polygon.astype('int32')], color=1)
        return cancer_mask


def test_vis():
    raw_data_dir = f"{os.getcwd()}/data/raw/ukt_breast"
    summary_dir = f"{os.getcwd()}/data/lmdb/ukt_breast"
    illustrator = Illustrator(raw_data_dir, processed_root_dir=summary_dir)
    with open('attention_scores.pickle', 'rb') as handle:
        attention_scores = pickle.load(handle)
    illustrator.save_attention_maps(root_dir=raw_data_dir, prediction=3, target=3, attention=attention_scores)


def plot_confmat(confmat, num_classes):
    df_cm = pd.DataFrame(confmat, index=range(1, num_classes+1), columns=range(1, num_classes+1))
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.05, right=.65)
    sn.set(font_scale=1.2)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax)

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', bbox_inches='tight')
    buf.seek(0)
    im = Image.open(buf)
    
    im = ToTensor()(im)
    return im


if __name__ == "__main__":
    import pickle

    test_vis()
