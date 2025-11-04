import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classifier.naive_bayes_classifier import NaiveBayesClassifier

class TestProgram:
    def __init__(self):
        self.unseen_food_texts = [
            "Grilled tofu with vegetables and quinoa",
            "Cheese pizza with mushrooms",
            "Chicken stir-fry with rice",
            "Oatmeal with almond milk and berries",
            "Bacon and eggs with toast"
        ]
        
        self.unseen_subject_texts = [
            "Por fracción molar, el aire seco contiene un 78% de nitrógen, un 20% de oxígeno, un 0.04% de dióxido de carbono y pequeñas cantidades de otros gases traza.",
            "La inflación afecta el poder adquisitivo de los consumidores.",
            "El cálculo diferencial estudia el cambio de funciones y sus derivadas.",
            "El PIB mide el crecimiento económico de un país.",
            "Las funciones trigonométricas son esenciales en la física y la ingeniería."
        ]

    def run_food_classification(self):
        with open("example1_food_classification.txt", "w", encoding="utf-8") as f:
            sys.stdout = f

            print("="*90)
            print("PROBLEM 1: FOOD CLASSIFICATION (Vegan, Vegetarian, Omnivorous)")
            print("="*90 + "\n")
            
            print("Unseen texts for classification:")
            for i, text in enumerate(self.unseen_food_texts, 1):
                print(f"{i}. {text}")

            # Excluding stop words
            print("\n---PROBLEM 1, EXAMPLE 1 EXCLUDING STOP WORDS ---\n")
            classifier = NaiveBayesClassifier(
                category_files={
                    'vegan': 'classifier/data/vegan.txt',
                    'vegetarian': 'classifier/data/vegetarian.txt',
                    'omnivorous': 'classifier/data/omnivorous.txt'
                },
                language='english',
                stop_words=True
            )
            freq = classifier.calculate_word_frequency()
            probability_categories = classifier.calculate_word_probability(freq)
            classifier.test_classify(self.unseen_food_texts, probability_categories)

            # Including stop words
            print("\n---PROBLEM 1, EXAMPLE 2 INCLUDING STOP WORDS ---\n")
            classifier0 = NaiveBayesClassifier(
                category_files={
                    'vegan': 'classifier/data/vegan.txt',
                    'vegetarian': 'classifier/data/vegetarian.txt',
                    'omnivorous': 'classifier/data/omnivorous.txt'
                },
                language='english',
                stop_words=False
            )
            freq0 = classifier0.calculate_word_frequency()
            probability_categories0 = classifier0.calculate_word_probability(freq0)
            classifier0.test_classify(self.unseen_food_texts, probability_categories0)

            print("\nNOTE: Removing stop words may slightly decrease classification effectiveness depending on the text.\n")
        
        sys.stdout = sys.__stdout__

    def run_subject_classification(self):
        with open("example2_subject_classification.txt", "w", encoding="utf-8") as f:
            sys.stdout = f

            print("="*90)
            print("PROBLEM 2: SUBJECT CLASSIFICATION (Biología, Matemáticas, Economía)")
            print("="*90 + "\n")
            
            print("Unseen texts for classification:")
            for i, text in enumerate(self.unseen_subject_texts, 1):
                print(f"{i}. {text}")

            # Excluding stop words
            print("\n---PROBLEMA 2, EJEMPLO 1 EXCLUDING STOP WORDS ---\n")
            classifier2 = NaiveBayesClassifier(
                category_files={
                    'biologia': 'classifier/data/biologia.txt',
                    'economia': 'classifier/data/economia.txt',
                    'matematicas': 'classifier/data/matematicas.txt'
                },
                language='spanish',
                stop_words=True
            )
            freq2 = classifier2.calculate_word_frequency()
            probability_categories2 = classifier2.calculate_word_probability(freq2)
            classifier2.test_classify(self.unseen_subject_texts, probability_categories2)

            # Including stop words
            print("\n---PROBLEMA 2, EJEMPLO 2 INCLUDING STOP WORDS ---\n")
            classifier1 = NaiveBayesClassifier(
                category_files={
                    'biologia': 'classifier/data/biologia.txt',
                    'economia': 'classifier/data/economia.txt',
                    'matematicas': 'classifier/data/matematicas.txt'
                },
                language='spanish',
                stop_words=False
            )
            freq1 = classifier1.calculate_word_frequency()
            probability_categories1 = classifier1.calculate_word_probability(freq1)
            classifier1.test_classify(self.unseen_subject_texts, probability_categories1)

            print("\nNOTE: Removing stop words may slightly reduce classification accuracy depending on the text.\n")
        
        sys.stdout = sys.__stdout__


if __name__ == "__main__":
    test_prog = TestProgram()
    test_prog.run_food_classification()
    test_prog.run_subject_classification()
