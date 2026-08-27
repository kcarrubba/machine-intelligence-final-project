# FoodTest1_expanded Image Classification Dataset

## Overview
This dataset was created for a deep learning project focused on classifying 40 different food items for a smart fridge application. It combines the provided course dataset (FoodTest1) with additional images from publicly available datasets and manually collected images to improve diversity and real-world robustness.

---

## Data Collection

### Provided Dataset
The base dataset was provided as part of the course materials and follows a structured naming convention:

**itemNumber_itemName_photoNumber.EXT**

---

### External Datasets
To expand coverage across all food classes, several publicly available datasets were used. Images were selectively extracted and mapped to the relevant categories.

- **Food-101 Dataset** (Bossard et al., 2014)
  Used for: Oyster, Steak, Cheese  
  License: Images sourced from Foodspotting; restricted to non-commercial scientific use
  http://www.foodspotting.com/

- **Freiburg Groceries Dataset** (Jund et al., 2016)
  Used for: Milk, Juice  
  License: Creative Commons Attribution 4.0 (CC BY 4.0)

- **Grocery Store Dataset** (Klasson et al., 2019)
  Used for: Asparagus, Mushrooms, Carrots, Garlic, Ginger, Capsicum, Tomato, Zucchini, Milk, Juice, Rockmelon, Watermelon, Avocado, Pineapple, Pear, Apple, Peach, Kiwi
  License: MIT License

- **A Large Scale Fish Dataset** (Ulucan et al., 2020)
  Used for: Prawn, Trout  
  License: Creative Commons Attribution 4.0 (CC BY 4.0)

- **Fruit Recognition Dataset** (Mureșan & Oltean, 2018)
  Used for: Carrot, Zucchini, Ginger, Capsicum, Rockmelon, Avocado, Tomato, Pear, Apple, Red Onion  
  License: Creative Commons Attribution 4.0 (CC BY 4.0)

- **Food for Machine Learning Dataset (FFML)** (Fulop & Cristea, 2020)
  Used for: Cheese, Chicken, Salmon  
  License: MIT License 

- **Fish-Vista Dataset** (Mehrab et al., 2025)
  Used for: Trout, Snapper
  License: MIT License
  Note: A subset of images was selected from the species classification data using scientific labels (e.g., Oncorhynchus mykiss, Pagrus auratus).

---

### Manual Data Collection
In addition to external datasets, over 6,000 images were manually captured by group members in real-world retail environments (e.g., Woolworths, Coles, and local fish markets).

Images were designed to reflect realistic usage conditions, including:
- Supermarket shelves, bins, and packaging
- Variations in lighting, angle, and distance
- Multiple instances of the same item

Care was taken to ensure:
- Each image focuses on a single food category
- Multiple different food types are not mixed within the same image
- A variety of perspectives and conditions are represented

This step was essential for improving robustness to real-world environments.

---

## Data Preprocessing

- File names were standardized to match the original dataset format:
**itemNumber_itemName_photoNumber.EXT**
- Original dataset images were preserved and not modified
- Additional images were renamed sequentially without overwriting existing files

---

## Usage Notes

- All datasets were used for **academic, non-commercial purposes only**
- All external datasets were used in accordance with their respective licenses:
  - Food-101: non-commercial scientific use only
  - Freiburg Groceries, Fruit Recognition, Fish Dataset: CC BY 4.0
  - Hierarchical Grocery, FFML, Fish-Vista: MIT License
- Proper attribution is provided in the project report and references

---

## Acknowledgements

We acknowledge and thank the creators of the datasets used in this project:
Bossard et al. (2014), Jund et al. (2016), Mureșan & Oltean (2018), Ulucan et al. (2020), Klasson et al. (2019), Fulop & Cristea (2020), and Mehrab et al. (2025), as well as the course staff for providing the base dataset.