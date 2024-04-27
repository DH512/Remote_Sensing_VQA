import os
import shutil
from typing import Any, Optional
import torch
import glob
import random
import PIL.Image as Image
import matplotlib.pyplot as plt
from cog import BasePredictor, Input, Path, BaseModel

from model.prismer_caption import PrismerCaption
from model.prismer_vqa import PrismerVQA
from dataset import create_dataset, create_loader
from dataset.utils import *
from utils import create_ade20k_label_colormap


class ModelOutput(BaseModel):
    answer: str
    depth: Optional[Path]
    edge: Optional[Path]
    segmentation: Optional[Path]
    surface_normal: Optional[Path]
    object_labels: Optional[Any]
    object_png: Optional[Path]
    ocr_pt: Optional[Path]
    ocr_png: Optional[Path]


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        os.system(
            f"python experts/segmentation/mask2former/modeling/pixel_decoder/ops/setup.py build install"
        )
        self.transform = Transform(
            resize_resolution=480, scale_size=[0.5, 1.0], train=False
        )
        self.obj_label_map = torch.load("dataset/detection_features.pt")["labels"]
        self.coco_label_map = torch.load("dataset/coco_features.pt")["labels"]
        self.ade_color = create_ade20k_label_colormap()

    def predict(
        self,
        input_image: Path = Input(
            description="Input image, only .png, .jpg, and .jpeg files are supported",
        ),
        task: str = Input(
            description="Choose between Visual Question Answering and Image Captioning",
            choices=["vqa", "caption"],
            default="caption",
        ),
        model_size: str = Input(
            description="Choose a model",
            choices=["base", "large"],
            default="base",
        ),
        use_experts: bool = Input(
            description="Load the experts if set to True",
            default=True,
        ),
        output_expert_labels: bool = Input(
            description="Return the experts output (when use_experts) if set to True",
            default=True,
        ),
        question: str = Input(
            description="Provide your question for Visual Question Answering task",
            default=None,
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""
        assert os.path.splitext(os.path.basename(str(input_image)))[-1] in [
            ".png",
            ".jpg",
            ".jpeg",
        ], "only .png, .jpg, and .jpeg files are supported"
        if task == "vqa":
            assert (
                question
            ), "Please provide a question for Visual Question Answering task."

        device = "cuda"

        model_name = (
            f"prismer_{model_size}" if use_experts else f"prismerz_{model_size}"
        )

        image = Image.open(str(input_image)).convert("RGB")
        helper_dir = "helpers"
        if use_experts:
            if os.path.exists(helper_dir):
                shutil.rmtree(helper_dir)
            os.makedirs(os.path.join(helper_dir, "images"))
            image.save(os.path.join(helper_dir, "images", "input.jpg"))

        config = {
            "prismer_model": model_name.replace("z", ""),
            "experts": [
                "depth",
                "normal",
                "seg_remo",
                "edge",
                "obj_detection",
                "ocr_detection",
            ]
            if use_experts
            else ["none"],
            "data_path": helper_dir if use_experts else None,
            "label_path": f"{helper_dir}/labels" if use_experts else None,
            "freeze": "freeze_vision",
            "image_resolution": 480,
            "prefix": "" if task == "vqa" else "A picture of",
            "dataset": "demo",
        }

        model = PrismerVQA(config) if task == "vqa" else PrismerCaption(config)
        state_dict = torch.load(
            f"weights/{task}_{model_name}/pytorch_model.bin", map_location=device
        )
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to(device)

        if use_experts:
            run_experts()
            _, test_dataset = create_dataset(
                "caption", config
            ) 
            test_loader = create_loader(
                test_dataset, batch_size=1, num_workers=4, train=False
            )
            experts, _ = next(iter(test_loader))
            experts = move_to_device(experts, device)
        else:
            experts = self.transform(image, None)
            experts["rgb"] = experts["rgb"].unsqueeze(0).to(device)

        with torch.no_grad():
            if task == "vqa":
                question = pre_question(question)
                result = model(experts, [question], train=False, inference="generate")
            else:
                question = pre_caption(config["prefix"])
                result = model(
                    experts, prefix=question, train=False, inference="generate"
                )

        if not (use_experts and output_expert_labels):
            return ModelOutput(answer=result[0])
        (
            depth,
            edge,
            segmentation,
            ocr_pt,
            ocr_png,
            surface_normal,
            obj_labels_dict,
            obj_png,
        ) = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

        labels_dir = f"{helper_dir}/labels"
        experts_results = os.listdir(labels_dir)
        if "edge" in experts_results:
            edge = "/tmp/edge.png"
            shutil.copyfile(
                os.path.join(
                    f"{labels_dir}/edge/helpers/images",
                    os.listdir(f"{labels_dir}/edge/helpers/images")[0],
                ),
                edge,
            )

        if "depth" in experts_results:
            label_path = os.path.join(
                f"{labels_dir}/depth/helpers/images",
                os.listdir(f"{labels_dir}/depth/helpers/images")[0],
            )
            depth_file = plt.imread(label_path)
            depth = "/tmp/depth.png"
            plt.imsave(depth, depth_file, cmap="rainbow")

        if "normal" in experts_results:
            surface_normal = "/tmp/normal.png"
            shutil.copyfile(
                os.path.join(
                    f"{labels_dir}/normal/helpers/images",
                    os.listdir(f"{labels_dir}/normal/helpers/images")[0],
                ),
                surface_normal,
            )

        if "seg_remo" in experts_results:
            label_path = os.path.join(
                f"{labels_dir}/seg_coco/helpers/images",
                os.listdir(f"{labels_dir}/seg_coco/helpers/images")[0],
            )
            rgb = plt.imread(str(input_image))
            seg_labels = plt.imread(label_path)
            plt.imshow(rgb)

            seg_map = np.zeros(list(seg_labels.shape) + [3], dtype=np.int16)
            for i in np.unique(seg_labels):
                seg_map[seg_labels == i] = self.ade_color[int(i * 255)]

            plt.imshow(seg_map, alpha=0.5)

            for i in np.unique(seg_labels):
                obj_idx_all = np.where(seg_labels == i)
                obj_idx = random.randint(0, len(obj_idx_all[0]))
                x, y = obj_idx_all[1][obj_idx], obj_idx_all[0][obj_idx]
                obj_name = self.coco_label_map[int(i * 255)]
                plt.text(
                    x,
                    y,
                    obj_name,
                    c="white",
                    horizontalalignment="center",
                    verticalalignment="center",
                )

            plt.axis("off")
            segmentation = "/tmp/segmentation.png"
            plt.savefig(
                segmentation, bbox_inches="tight", transparent=True, pad_inches=0
            )
            plt.close()

        if "ocr_detection" in experts_results:
            label_path = [
                file
                for file in os.listdir(f"{labels_dir}/ocr_detection/helpers/images")
                if file.endswith(".png")
            ][0]
            ocr_labels = plt.imread(
                os.path.join(f"{labels_dir}/ocr_detection/helpers/images", label_path)
            )

            pt_path = f"{os.path.splitext(os.path.basename(label_path))[0]}.pt"
            ocr_labels_dict = torch.load(
                os.path.join(f"{labels_dir}/ocr_detection/helpers/images", pt_path)
            )
            rgb = plt.imread(str(input_image))
            plt.imshow(rgb)
            plt.imshow((1 - ocr_labels) < 1, cmap="gray", alpha=0.8)

            ocr_pt = "/tmp/ocr.pt"
            ocr_png = "/tmp/ocr.png"

            for i in np.unique(ocr_labels)[:-1]:
                text_idx_all = np.where(ocr_labels == i)
                x, y = text_idx_all[1].mean(), text_idx_all[0].mean()
                text = ocr_labels_dict[int(i * 255)]["text"]
                plt.text(
                    x,
                    y,
                    text,
                    c="white",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
            plt.axis("off")
            plt.savefig(ocr_png, bbox_inches="tight", transparent=True, pad_inches=0)
            plt.close()
            shutil.copyfile(
                os.path.join(f"{labels_dir}/ocr_detection/helpers/images", pt_path),
                ocr_pt,
            )

        if "obj_detection" in experts_results:
            label_path = [
                file
                for file in os.listdir(f"{labels_dir}/obj_detection/helpers/images")
                if file.endswith(".png")
            ][0]
            obj_labels = plt.imread(
                os.path.join(f"{labels_dir}/obj_detection/helpers/images", label_path)
            )
            json_path = f"{os.path.splitext(os.path.basename(label_path))[0]}.json"
            obj_labels_dict = json.load(
                open(
                    os.path.join(
                        f"{labels_dir}/obj_detection/helpers/images", json_path
                    )
                )
            )
            rgb = plt.imread(str(input_image))
            plt.imshow(rgb)
            num_objs = np.unique(obj_labels)[:-1].max()
            plt.imshow(obj_labels, cmap="terrain", vmax=num_objs + 1 / 255.0, alpha=0.5)

            for i in np.unique(obj_labels)[:-1]:
                obj_idx_all = np.where(obj_labels == i)
                obj_idx = random.randint(0, len(obj_idx_all[0]))
                x, y = obj_idx_all[1][obj_idx], obj_idx_all[0][obj_idx]
                obj_name = self.obj_label_map[obj_labels_dict[str(int(i * 255))]]
                plt.text(
                    x,
                    y,
                    obj_name,
                    c="white",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
            plt.axis("off")
            obj_png = "/tmp/obj.png"
            plt.savefig(obj_png, bbox_inches="tight", transparent=True, pad_inches=0)
            plt.close()

        return ModelOutput(
            answer=result[0],
            depth=Path(depth) if depth else None,
            edge=Path(edge) if edge else None,
            segmentation=Path(segmentation) if segmentation else None,
            ocr_png=Path(ocr_png) if ocr_png else None,
            ocr_pt=Path(ocr_pt) if ocr_pt else None,
            surface_normal=Path(surface_normal) if surface_normal else None,
            object_labels=obj_labels_dict,
            object_png=Path(obj_png) if obj_png else None,
        )


def run_experts():
    experts = ["edge", "depth", "normal", "objdet", "ocrdet", "segmentation"]
    for expert in experts:
        print(f"***** Generating {expert} *****")
        os.system(f"python experts/generate_{expert}.py")

