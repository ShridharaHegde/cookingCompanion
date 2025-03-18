# from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import ollama

import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama

def extract_text_from_pdfs(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            documents.append({"filename": filename, "content": text})
    return documents

def create_index(documents):
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    index = []
    for doc in documents:
        chunks = [doc["content"][i:i+500] for i in range(0, len(doc["content"]), 500)]
        embeddings = embedding_model.encode(chunks)
        index.extend([{"chunk": chunk, "embedding": embedding} for chunk, embedding in zip(chunks, embeddings)])
    return index, embedding_model

def retrieve_relevant_chunks(query, index, embedding_model, top_k=3):
    query_embedding = embedding_model.encode([query])
    similarities = [cosine_similarity([query_embedding[0]], [entry["embedding"]])[0][0] for entry in index]
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
    return [index[i]["chunk"] for i in top_indices]

def augment_prompt_with_context(query, retrieved_chunks):
    context = "\n".join(retrieved_chunks)
    prompt = f"Use the following recipes to answer the question:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    return prompt

def call_to_mistral_with_rag(query, index, embedding_model):
    retrieved_chunks = retrieve_relevant_chunks(query, index, embedding_model)
    augmented_prompt = augment_prompt_with_context(query, retrieved_chunks)

    stream = ollama.generate(model='mistral', prompt=augmented_prompt,
                             stream=True, options={"temperature": 0.8, "top_p": 0.9, "max_tokens": 4096})
    for chunk in stream:
        if 'response' in chunk:
            yield chunk['response']


def analyze_image(image):
    # model = YOLO('yolov8n.pt')
    # results = model(image, conf=0.3)
    #
    # result = results[0]

    # img = cv2.imread(image)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    # fig, ax = plt.subplots(figsize=(12, 8))
    # ax.imshow(result.plot())
    # ax.axis('off')
    # plt.show()

    # classes = []
    #
    # for box in result.boxes:
    #     class_id = result.names[box.cls[0].item()]
    #     classes.append(class_id)
    
    # return classes

    im = cv2.imread("images/fridge0.png")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    cfg.MODEL.DEVICE = "cpu"

    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1],
                   MetadataCatalog.get(cfg.DATASETS.TRAIN[0] if len(cfg.DATASETS.TRAIN) else "__unused"),
                   scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    result = v.get_image()[:, :, ::-1]
    # cv2.imshow("Detected Objects", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0] if len(cfg.DATASETS.TRAIN) else "__unused")
    class_ids = outputs["instances"].pred_classes.cpu().numpy()
    class_labels = [metadata.thing_classes[i] for i in class_ids]
    print("Detected Objects:", class_labels)

    return list(set(class_labels))

#
# def call_to_mistral(ingredients):
#     stream = ollama.generate(model='mistral', prompt=f"What can I cook using these ingredients assuming I have other basic ingredients such as oil, salt, water, etc {ingredients}", stream=True)
#     for chunk in stream:
#         if 'response' in chunk:
#             yield chunk['response']

def driver(image):
    classes = analyze_image(image)
    recipes_directory = "Recipes"

    recipes = extract_text_from_pdfs(recipes_directory)

    index, embedding_model = create_index(recipes)

    user_query = f"Elaborate recipie(s) in great detail enumerating ingredients and step by step instructions with the provided ingredients: {classes} in a markdown format, assume that basic cooking essentials such as salt, masala powders are available. If some of these are not ingredients ignore them."

    return call_to_mistral_with_rag(user_query, index, embedding_model)


# image_path = '/Users/shridharahegde/Desktop/ML Projects/Image to recepie/images/fridge0.png'
# res = analyze_image(image_path)
# res = str(res)
# call_to_mistral(res)


