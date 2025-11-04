import sys
import os

# Ensure the project root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classifier.naive_bayes_classifier import NaiveBayesClassifier


class TestProgramDynamic:
    def __init__(self):
        # Define file paths for category training data
        self.food_categories = {
            'vegan': 'classifier/data/vegan_dynamic.txt',
            'vegetarian': 'classifier/data/vegetarian_dynamic.txt',
            'omnivorous': 'classifier/data/omnivorous_dynamic.txt'
        }

        self.subject_categories = {
            'biologia': 'classifier/data/biologia_dynamic.txt',
            'economia': 'classifier/data/economia_dynamic.txt',
            'matematicas': 'classifier/data/matematicas_dynamic.txt'
        }

    def run(self):
        print("=" * 90)
        print("DYNAMIC TEXT CLASSIFIER (Stop Words Removed)")
        print("=" * 90)
        print("This program classifies text into one of two groups:\n")
        print("1. Food type (vegan / vegetarian / omnivorous)")
        print("2. Academic subject (biología / economía / matemáticas)")
        print("\nType 'exit' anytime to quit.\n")

        while True:
            print("-" * 90)
            mode = input("Please type 1 or 2 and press Enter (1 = Food, 2 = Subject): ").strip()

            if mode.lower() == 'exit':
                print("\nExiting program. Goodbye!")
                break

            if mode not in ['1', '2']:
                print("Invalid option. Please enter 1 or 2.\n")
                continue

            print("\nNow enter a sentence to classify (or type 'exit' to quit):")
            text = input("Your text: ").strip()

            if text.lower() == 'exit':
                print("\nExiting program. Goodbye!")
                break

            if not text:
                print("Please enter a non-empty sentence.\n")
                continue

            if mode == '1':
                self.classify_text(
                    text=text,
                    category_files=self.food_categories,
                    language='english',
                    title="FOOD CLASSIFICATION"
                )
            else:
                self.classify_text(
                    text=text,
                    category_files=self.subject_categories,
                    language='spanish',
                    title="SUBJECT CLASSIFICATION"
                )

    def classify_text(self, text, category_files, language, title):
        print("\n" + "=" * 90)
        print(f"{title} (Stop Words Removed)")
        print("=" * 90)
        print(f"Text to classify: {text}\n")

        # Initialize the classifier (always removing stop words)
        classifier = NaiveBayesClassifier(
            category_files=category_files,
            language=language,
            stop_words=True
        )

        # Train and classify
        freq = classifier.calculate_word_frequency()
        probs = classifier.calculate_word_probability(freq)
        classifier.test_classify([text], probs)

        print("\nNote: Stop words were automatically removed for higher accuracy.\n")


if __name__ == "__main__":
    program = TestProgramDynamic()
    program.run()