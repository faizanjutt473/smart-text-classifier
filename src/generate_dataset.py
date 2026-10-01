"""
generate_dataset.py
-------------------
Builds a labelled text-classification dataset with four categories:
Complaint, Inquiry, Feedback, Other.

How the dataset is created
    1. A small set of hand-written samples per class (including tricky,
       borderline sentences that make the task more realistic).
    2. Many additional samples produced from sentence templates combined with
       varied slot values (products, cities, issues, features, ...).
    3. Light "real-world noise" (lower-casing, missing punctuation) is applied
       to a share of the samples so the model sees messy text as well.
    4. Duplicates are removed and every class is balanced to the same size.

The process is seeded, so running the script again gives the same CSV.

Usage:
    python src/generate_dataset.py
"""

import random
from pathlib import Path

import pandas as pd

SEED = 42
SAMPLES_PER_CLASS = 300
NOISE_RATE = 0.15
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "text_classification_dataset.csv"

rng = random.Random(SEED)

# --------------------------------------------------------------------------
# Shared slot values
# --------------------------------------------------------------------------
CITIES = ["Lahore", "Karachi", "Islamabad", "Faisalabad", "Multan", "Peshawar", "Quetta", "Sialkot"]
PRODUCTS = ["laptop", "smartphone", "headphones", "water bottle", "office chair", "tablet", "smart watch", "backpack"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
TIMES = ["9 am", "10:30 am", "12 pm", "2 pm", "3:30 pm", "5 pm"]
NAMES = ["Ali", "Sara", "Ahmed", "Fatima", "Usman", "Ayesha", "Bilal", "Hina", "Hamza", "Zainab"]


def pick(options):
    return rng.choice(options)


# --------------------------------------------------------------------------
# COMPLAINT
# --------------------------------------------------------------------------
COMPLAINT_HANDWRITTEN = [
    "I have been waiting for my refund for three weeks and nobody is responding.",
    "The product stopped working after two days. This is a waste of money.",
    "Your delivery guy was rude and left the parcel outside in the rain.",
    "I was charged twice for the same order, please reverse the extra payment.",
    "The app crashes every time I try to open my profile page.",
    "I ordered a blue jacket but received a red one, and the zipper is broken.",
    "Support promised a callback yesterday and never called. Very unprofessional.",
    "My account was locked for no reason and I cannot access my files.",
    "The internet has been down for four days and no technician has visited.",
    "Worst service ever, I will never order from you again.",
    "The invoice shows a wrong amount and I was billed for items I never bought.",
    "I returned the item a month ago and still have not received my money back.",
]
COMPLAINT_OPENERS = [
    "I am very disappointed with the fact that",
    "I am writing to complain because",
    "This is unacceptable:",
    "I am extremely frustrated because",
    "I want to report a serious problem:",
    "I am not happy at all because",
    "Terrible experience:",
    "Once again I have an issue:",
]
COMPLAINT_ISSUES = [
    "my order arrived two weeks late",
    "the product I received was damaged",
    "the app keeps crashing whenever I log in",
    "I was charged twice for the same subscription",
    "customer support has not replied to my emails for a week",
    "the refund I was promised has not been processed",
    "the wrong item was delivered to my address",
    "the internet connection drops every few minutes",
    "the website is extremely slow and unusable",
    "my account was suspended without any notice",
    "the package was missing two items",
    "the {product} I bought stopped working within a week",
    "the delivery to {city} has been delayed again",
    "the staff at your branch in {city} treated me badly",
    "the payment was deducted but my order was never confirmed",
    "the quality of the {product} is far worse than advertised",
    "I cannot log in even after resetting my password three times",
    "the discount code was rejected at checkout even though it is valid",
    "the live chat agent kept disconnecting me without solving anything",
    "the service quality has become very poor lately",
]
COMPLAINT_DEMANDS = [
    "Please fix this immediately.",
    "I expect a full refund.",
    "This needs urgent attention.",
    "I want this resolved as soon as possible.",
    "Please escalate this to a manager.",
    "I will cancel my service if this continues.",
    "Nobody has helped me so far.",
    "",
]

# --------------------------------------------------------------------------
# INQUIRY
# --------------------------------------------------------------------------
INQUIRY_HANDWRITTEN = [
    "What are your working hours on weekends?",
    "How can I change the email address linked to my account?",
    "Do you offer a student discount on the annual plan?",
    "Could you tell me how long international shipping usually takes?",
    "Is the premium plan available for teams of more than ten people?",
    "I am unhappy with the price, but can you tell me if a cheaper plan exists?",
    "Where can I download the invoice for my last purchase?",
    "Which documents do I need to open a business account?",
    "When will the new version of the app be released?",
    "Can I upgrade my subscription in the middle of a billing cycle?",
    "Does the warranty cover accidental damage?",
    "Please let me know the steps to register for the next batch.",
]
INQUIRY_OPENERS = [
    "Could you please tell me",
    "I would like to know",
    "Can you let me know",
    "Is it possible to find out",
    "Do you know",
    "I wanted to ask",
    "Please let me know",
    "Kindly inform me",
]
INQUIRY_TOPICS = [
    "what your business hours are",
    "how I can reset my password",
    "whether you deliver to {city}",
    "what the price of the premium plan is",
    "how long delivery usually takes",
    "which payment methods you accept",
    "if there is a discount for first-time customers",
    "how to track my order",
    "where I can find my invoice",
    "what documents are needed to open an account",
    "whether the {product} comes with a warranty",
    "how I can update my billing address",
    "when the new version will be released",
    "if I can upgrade my plan later",
    "how to contact the support team",
    "what the return policy is",
    "whether the service is available on Android",
    "how I can cancel my subscription",
    "if you offer bulk discounts",
    "when registration for the next batch opens",
    "whether the {product} is available in stock",
    "if you have a branch in {city}",
]
INQUIRY_DIRECT = [
    "How much does the {plan} plan cost?",
    "Do you deliver to {city}?",
    "Is the {product} available in black?",
    "What is the delivery time for {city}?",
    "How do I return a {product}?",
    "Does the {plan} plan include phone support?",
    "Can I pay for the {product} in instalments?",
    "What is the difference between the {plan} and {plan2} plans?",
    "Is there a free trial for the {plan} plan?",
    "How can I get a receipt for my {product} purchase?",
]
PLANS = ["Basic", "Standard", "Premium", "Business", "Enterprise"]

# --------------------------------------------------------------------------
# FEEDBACK
# --------------------------------------------------------------------------
FEEDBACK_HANDWRITTEN = [
    "I really like the new dashboard, it is clean and easy to use.",
    "The delivery was quick and the packaging was neat. Good job.",
    "It would be nice to have a dark mode option in the app.",
    "Overall a good experience, though the checkout page could be simpler.",
    "The support agent was friendly and explained everything clearly.",
    "I suggest adding more payment options such as mobile wallets.",
    "Great quality for the price. I would recommend it to my friends.",
    "The tutorial videos are helpful, maybe add some advanced ones as well.",
    "The new update is smoother than the old version, nice work.",
    "The product is decent, but a longer battery life would make it perfect.",
    "Loved the workshop, the instructor was knowledgeable and patient.",
    "The interface looks professional. Adding a search bar would improve it further.",
]
FEATURES = [
    "a dark mode",
    "an offline mode",
    "more payment options",
    "a mobile version of the website",
    "a better search function",
    "more tutorial videos",
    "a wish-list feature",
    "order tracking notifications",
    "support for Urdu language",
    "a one-click reorder button",
    "a product comparison tool",
    "a loyalty rewards program",
    "weekly summary emails",
    "a live chat option on the homepage",
    "bigger buttons for easier tapping",
    "a faster way to export reports",
]
PRAISE_TOPICS = [
    "delivery speed",
    "packaging",
    "customer service",
    "user interface",
    "product quality",
    "checkout process",
    "mobile app",
    "website design",
    "onboarding experience",
    "documentation",
    "return process",
    "search page",
    "order confirmation email",
    "product range",
    "pricing structure",
    "training material",
]
ADJECTIVES = [
    "friendly", "helpful", "professional", "polite", "knowledgeable", "patient",
    "cooperative", "attentive",
]
FEEDBACK_TEMPLATES = [
    "I really like the {praise}. Keep up the good work!",
    "The {praise} was excellent and exceeded my expectations.",
    "Overall I am satisfied with the {praise}.",
    "It would be great if you could add {feature}.",
    "My suggestion is to include {feature} in a future update.",
    "The staff was very {adj} and I enjoyed the experience.",
    "Thank you for the great {praise}, it was a pleasant experience.",
    "The {praise} is good, but there is some room for improvement in the {praise2}.",
    "I think {feature} would make the app even better.",
    "Five stars for the {praise}. Highly recommended.",
    "Nice work on the {praise}, the team should be proud.",
    "A small suggestion: please consider {feature}.",
    "I am happy with the {praise}, well done.",
    "The {praise} has improved a lot since last year, thank you.",
    "Just wanted to say that the {praise} was really smooth this time.",
    "Good experience overall. Adding {feature} would be a nice bonus.",
    "The {praise} is great, and the {adj} team made it even better.",
    "I would love to see {feature} in the next release.",
    "Honest feedback: the {praise} is good but {feature} would be a big improvement.",
    "Impressed by the {praise}. Please keep it up.",
]

# --------------------------------------------------------------------------
# OTHER  (greetings, announcements, spam, small talk, acknowledgements)
# --------------------------------------------------------------------------
OTHER_HANDWRITTEN = [
    "Good morning everyone.",
    "Hello, hope you are doing well.",
    "Test message, please ignore.",
    "Congratulations! You have won a free vacation. Click the link to claim now.",
    "Happy Eid to you and your family.",
    "The weather is lovely today.",
    "Noted, thanks.",
    "asdfgh qwerty",
    "Earn 5000 rupees per day working from home, no experience needed.",
    "Forwarded as requested.",
    "I am out of office and will reply when I return.",
    "See you at the event tomorrow.",
]
OTHER_GREETINGS = [
    "Hello {name}, hope you are doing well.",
    "Good morning {name}.",
    "Hi {name}, how are you?",
    "Good evening {name}, have a nice day.",
    "Assalam o Alaikum {name}.",
    "Hey {name}, long time no see.",
    "Happy birthday {name}, enjoy your day.",
    "Take care {name}, talk to you soon.",
]
OTHER_ANNOUNCEMENTS = [
    "The team meeting is scheduled for {day} at {time}.",
    "Reminder: the training session starts on {day} at {time}.",
    "The office will remain closed on {day}.",
    "I will be out of office on {day} and will reply when I return.",
    "The webinar has been moved to {day} at {time}.",
    "Lunch will be served at {time} in the main hall.",
]
OTHER_MISC = [
    "Noted, thank you.",
    "Okay, got it.",
    "Forwarded for your information.",
    "This is an automated message, please do not reply.",
    "Your one-time code is {code}. Do not share it with anyone.",
    "Congratulations! You have won a {prize}. Click the link to claim it now.",
    "Limited offer: buy one get one free on all {product}s, click here!",
    "Earn {amount} rupees daily from home with no experience.",
    "The weather in {city} is very pleasant today.",
    "Did you watch the cricket match last night?",
    "Attached is the file we discussed earlier.",
    "Thanks for the update.",
    "Please ignore my previous message.",
    "Sent from my phone.",
]
PRIZES = ["brand new phone", "free vacation", "gift voucher", "lucky draw car"]


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------
def fill(template):
    """Replace every slot in a template with a random value."""
    values = {
        "city": pick(CITIES),
        "product": pick(PRODUCTS),
        "day": pick(DAYS),
        "time": pick(TIMES),
        "name": pick(NAMES),
        "plan": pick(PLANS),
        "plan2": pick(PLANS),
        "feature": pick(FEATURES),
        "praise": pick(PRAISE_TOPICS),
        "praise2": pick(PRAISE_TOPICS),
        "adj": pick(ADJECTIVES),
        "code": str(rng.randint(100000, 999999)),
        "prize": pick(PRIZES),
        "amount": str(rng.choice([2000, 3000, 5000, 10000])),
    }
    return template.format(**values)


def clean_spaces(text):
    return " ".join(text.split())


def complaint_sample():
    issue = fill(pick(COMPLAINT_ISSUES))
    demand = pick(COMPLAINT_DEMANDS)
    # Real complaints often state the problem directly, without any opener,
    # so about 40% of the samples are plain problem statements.
    if rng.random() < 0.4:
        sentence = f"{issue}."
    else:
        sentence = f"{pick(COMPLAINT_OPENERS)} {issue}."
    sentence = sentence[0].upper() + sentence[1:]
    return clean_spaces(f"{sentence} {demand}")


def inquiry_sample():
    if rng.random() < 0.4:
        return clean_spaces(fill(pick(INQUIRY_DIRECT)))
    return clean_spaces(f"{pick(INQUIRY_OPENERS)} {fill(pick(INQUIRY_TOPICS))}?")


def feedback_sample():
    return clean_spaces(fill(pick(FEEDBACK_TEMPLATES)))


def other_sample():
    group = rng.random()
    if group < 0.35:
        return clean_spaces(fill(pick(OTHER_GREETINGS)))
    if group < 0.65:
        return clean_spaces(fill(pick(OTHER_ANNOUNCEMENTS)))
    return clean_spaces(fill(pick(OTHER_MISC)))


def add_noise(text):
    """Make a share of samples look like real, messy user input."""
    if rng.random() >= NOISE_RATE:
        return text
    if rng.random() < 0.5:
        return text.lower()
    return text.rstrip(".?!")


def build_class(handwritten, generator):
    samples = list(handwritten)
    seen = set(s.lower() for s in samples)
    attempts = 0
    while len(samples) < SAMPLES_PER_CLASS and attempts < 100000:
        attempts += 1
        text = generator()
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        samples.append(text)
    if len(samples) < SAMPLES_PER_CLASS:
        raise RuntimeError("Could not generate enough unique samples; add more templates.")
    # Noise is applied after de-duplication, but only to generated samples
    hand = samples[: len(handwritten)]
    generated = [add_noise(s) for s in samples[len(handwritten):]]
    return hand + generated


def main():
    data = []
    for label, handwritten, generator in [
        ("Complaint", COMPLAINT_HANDWRITTEN, complaint_sample),
        ("Inquiry", INQUIRY_HANDWRITTEN, inquiry_sample),
        ("Feedback", FEEDBACK_HANDWRITTEN, feedback_sample),
        ("Other", OTHER_HANDWRITTEN, other_sample),
    ]:
        for text in build_class(handwritten, generator):
            data.append({"text": text, "category": label})

    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset="text").sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.insert(0, "id", range(1, len(df) + 1))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(df["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
