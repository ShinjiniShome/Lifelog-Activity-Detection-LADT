import pandas as pd

u1_visual_concepts = pd.read_csv("Dataset/visual_concepts/u1_categories_attr_concepts.csv")
u1_activity_annotation = pd.read_excel("Dataset/ttqtran_u1_categories_attr_concepts.xlsx", 'ttqtran_u1_categories_attr_conc')

#Sync all data at one place based on image id
training_data = u1_visual_concepts.merge(u1_activity_annotation[["image_id","ADL", "ENV"]], on="image_id", how="inner")

#Get combinations of ADL and ENV
training_data["ADL_ENV"] = training_data["ADL"].replace(r"^ +| +$", r"", regex=True) + " " + training_data["ENV"].replace(r"^ +| +$", r"", regex=True)

#Get unique combinations of ADL and ENV to manually assign official classes to
#unique_ADL_ENV = training_data["ADL_ENV"].unique()
#unique_ADL_ENV.to_excel("Dataset/ADL_ENV.xlsx")

#ADL_ENV file contains just the unique combinations of ADL and ENV attributes
ADL_ENV_file = pd.read_excel("TempData/ADL_ENV.xlsx")

#Official_Classes.xlsx file contains just the official classes that are manually assigned to unique combinations of ADL and ENV
official_classes = pd.read_excel("TempData/Official_Classes.xlsx")

ADL_ENV_official_classes = pd.DataFrame()
ADL_ENV_official_classes["ADL_ENV"] = ADL_ENV_file["ADL_ENV"]
ADL_ENV_official_classes["Official_Classes"] = official_classes["Official_Classes"]

#assign official classes to all the training data based on common ADL_ENV combination
training_data_with_classes = training_data.merge(ADL_ENV_official_classes[["ADL_ENV","Official_Classes"]], on="ADL_ENV", how="inner")
training_data_with_classes.to_excel("TempData/training_data_with_classes.xlsx", index = False)