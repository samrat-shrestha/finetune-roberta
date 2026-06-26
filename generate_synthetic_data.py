"""
generate_synthetic_data.py
--------------------------
Generates synthetic clinical notes with injected PHI and outputs JSONL
with character-level spans in OBI ehr_deidentification format.

Output format (one JSON object per line):
{
  "text": "Patient John Smith was seen on 03/12/1989...",
  "spans": [
    {"start": 8, "end": 18, "label": "PATIENT", "text": "John Smith"},
    {"start": 30, "end": 40, "label": "DATE",    "text": "03/12/1989"}
  ]
}

Usage:
  python generate_synthetic_data.py --num_notes 500 --output data/synthetic_train.jsonl
  python generate_synthetic_data.py --num_notes 100 --output data/synthetic_val.jsonl --seed 99
"""

import json
import logging
import random
import argparse
import os
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PHI pools — realistic but entirely fake
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "James", "Maria", "Robert", "Linda", "Michael", "Patricia", "William",
    "Barbara", "David", "Susan", "Richard", "Jessica", "Joseph", "Sarah",
    "Thomas", "Karen", "Charles", "Lisa", "Christopher", "Nancy", "Daniel",
    "Betty", "Matthew", "Margaret", "Anthony", "Sandra", "Donald", "Ashley",
    "Mark", "Dorothy", "Paul", "Kimberly", "Steven", "Emily", "Andrew", "Donna",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
]

DOCTOR_PREFIXES = ["Dr.", "Dr"]

STREETS = [
    "Main St", "Oak Ave", "Maple Dr", "Cedar Ln", "Pine Rd", "Elm St",
    "Washington Blvd", "Park Ave", "Lake Dr", "River Rd", "Hill Ct",
    "Forest Way", "Sunset Blvd", "Highland Ave", "Meadow Ln",
]

CITIES = [
    "New Orleans", "Baton Rouge", "Shreveport", "Metairie", "Kenner",
    "Covington", "Hammond", "Lafayette", "Lake Charles", "Monroe",
    "Houston", "Dallas", "Austin", "San Antonio", "Memphis",
]

STATES = ["LA", "TX", "MS", "AL", "TN", "GA", "FL"]

HOSPITALS = [
    "Tulane Medical Center", "University Medical Center",
    "Children's Hospital New Orleans", "Ochsner Medical Center",
    "LSU Health Sciences Center", "East Jefferson General Hospital",
    "St. Tammany Parish Hospital", "Our Lady of the Lake Regional Medical Center",
]

PHONE_FORMATS = [
    "{a}-{b}-{c}",
    "({a}) {b}-{c}",
    "{a}.{b}.{c}",
]

EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "tulane.edu", "lsuhsc.edu"]

ID_FORMATS = [
    "MRN{n}",
    "PT-{n}",
    "{n}",
]

CHIEF_COMPLAINTS = [
    "chest pain", "shortness of breath", "abdominal pain", "headache",
    "fever and chills", "nausea and vomiting", "dizziness", "back pain",
    "fatigue", "cough", "palpitations", "leg swelling", "rash",
    "joint pain", "difficulty breathing",
]

DIAGNOSES = [
    "Type 2 diabetes mellitus", "Hypertension", "Congestive heart failure",
    "Community-acquired pneumonia", "Acute myocardial infarction",
    "Chronic kidney disease stage 3", "Asthma exacerbation",
    "Urinary tract infection", "Deep vein thrombosis",
    "Atrial fibrillation", "GERD", "Anxiety disorder",
]

MEDICATIONS = [
    "metformin 500mg BID", "lisinopril 10mg daily", "atorvastatin 40mg QHS",
    "aspirin 81mg daily", "metoprolol succinate 25mg daily",
    "omeprazole 20mg daily", "amlodipine 5mg daily", "furosemide 40mg daily",
]

# ---------------------------------------------------------------------------
# Generators for individual PHI values
# ---------------------------------------------------------------------------

def random_name(rng):
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"

def random_doctor_name(rng):
    prefix = rng.choice(DOCTOR_PREFIXES)
    return f"{prefix} {rng.choice(LAST_NAMES)}"

def random_date(rng):
    """Return a random date string in one of several realistic formats."""
    base = datetime(1940, 1, 1) + timedelta(days=rng.randint(0, 30000))
    fmt = rng.choice([
        "%m/%d/%Y",   # 03/12/1989
        "%m-%d-%Y",   # 03-12-1989
        "%B %d, %Y",  # March 12, 1989
        "%b %d, %Y",  # Mar 12, 1989
        "%m/%d/%y",   # 03/12/89
        "%Y-%m-%d",   # 1989-03-12
    ])
    return base.strftime(fmt)

def random_phone(rng):
    a = rng.randint(200, 999)
    b = rng.randint(200, 999)
    c = rng.randint(1000, 9999)
    fmt = rng.choice(PHONE_FORMATS)
    return fmt.format(a=a, b=b, c=c)

def random_address(rng):
    num = rng.randint(100, 9999)
    street = rng.choice(STREETS)
    city = rng.choice(CITIES)
    state = rng.choice(STATES)
    zip_code = rng.randint(10000, 99999)
    return f"{num} {street}, {city}, {state} {zip_code}"

def random_age(rng):
    return str(rng.randint(1, 95))

def random_mrn(rng):
    n = rng.randint(100000, 9999999)
    fmt = rng.choice(ID_FORMATS)
    return fmt.format(n=n)

def random_email(rng):
    first = rng.choice(FIRST_NAMES).lower()
    last = rng.choice(LAST_NAMES).lower()
    domain = rng.choice(EMAIL_DOMAINS)
    sep = rng.choice([".", "_", ""])
    num = rng.choice(["", str(rng.randint(1, 99))])
    return f"{first}{sep}{last}{num}@{domain}"

def random_hospital(rng):
    return rng.choice(HOSPITALS)

# ---------------------------------------------------------------------------
# Note template engine
# ---------------------------------------------------------------------------

def inject(text, value, label, spans):
    """
    Find `value` in `text` and record its span.
    Returns updated text (no change) and updated spans list.
    We search from the end of the last span to avoid duplicate matches.
    """
    start_search = spans[-1]["end"] if spans else 0
    idx = text.find(value, start_search)
    if idx == -1:
        # fallback: search from beginning (value may appear before last span)
        idx = text.find(value)
    if idx != -1:
        spans.append({
            "start": idx,
            "end": idx + len(value),
            "label": label,
            "text": value,
        })
    return spans


def build_note(rng):
    """
    Build one synthetic clinical note. Returns (text, spans).
    Spans are character-level, non-overlapping, sorted by start position.
    """
    # --- Generate PHI values for this note ---
    patient_name   = random_name(rng)
    doctor_name    = random_doctor_name(rng)
    dob            = random_date(rng)
    visit_date     = random_date(rng)
    discharge_date = random_date(rng)
    phone          = random_phone(rng)
    address        = random_address(rng)
    age            = random_age(rng)
    mrn            = random_mrn(rng)
    email          = random_email(rng)
    hospital       = random_hospital(rng)
    cc             = rng.choice(CHIEF_COMPLAINTS)
    dx             = rng.choice(DIAGNOSES)
    med            = rng.choice(MEDICATIONS)

    # --- Pick a template ---
    templates = [

        # Template 1: Discharge summary style
        (
            f"DISCHARGE SUMMARY\n\n"
            f"Patient: {patient_name}\n"
            f"MRN: {mrn}\n"
            f"Date of Birth: {dob}\n"
            f"Admission Date: {visit_date}\n"
            f"Discharge Date: {discharge_date}\n"
            f"Attending Physician: {doctor_name}\n"
            f"Hospital: {hospital}\n\n"
            f"CHIEF COMPLAINT:\n"
            f"The patient is a {age}-year-old presenting with {cc}.\n\n"
            f"HISTORY OF PRESENT ILLNESS:\n"
            f"{patient_name} is a {age}-year-old who presented to {hospital} "
            f"on {visit_date} with complaints of {cc}. "
            f"The patient has a history of {dx}. "
            f"Contact phone: {phone}. Home address: {address}.\n\n"
            f"ASSESSMENT AND PLAN:\n"
            f"Diagnosis: {dx}.\n"
            f"Start {med}. Follow up in 2 weeks.\n"
            f"Patient email on file: {email}.\n"
            f"Signed, {doctor_name}\n"
        ),

        # Template 2: Progress note style
        (
            f"PROGRESS NOTE — {visit_date}\n\n"
            f"Name: {patient_name}   DOB: {dob}   MRN: {mrn}\n"
            f"Attending: {doctor_name}\n"
            f"Facility: {hospital}\n\n"
            f"S: Patient reports {cc}. Age {age}. "
            f"Reached at {phone} this morning to confirm appointment.\n\n"
            f"O: Vitals stable. Physical exam unremarkable.\n\n"
            f"A: {dx}.\n\n"
            f"P: Continue {med}. "
            f"Patient lives at {address}. "
            f"Email correspondence: {email}. "
            f"Return to clinic on {discharge_date}.\n"
            f"—{doctor_name}\n"
        ),

        # Template 3: Referral letter style
        (
            f"To: Consulting Physician\n"
            f"From: {doctor_name}\n"
            f"Date: {visit_date}\n"
            f"Re: {patient_name}, DOB {dob}, MRN {mrn}\n\n"
            f"Dear Colleague,\n\n"
            f"I am writing to refer {patient_name}, a {age}-year-old patient "
            f"currently followed at {hospital}, for further evaluation of {cc}. "
            f"The patient carries a diagnosis of {dx} and is currently on {med}.\n\n"
            f"Please feel free to contact me at {phone} or {email}.\n"
            f"The patient's address is {address}.\n\n"
            f"Thank you for your assistance.\n\n"
            f"Sincerely,\n{doctor_name}\n{hospital}\n"
        ),

        # Template 4: ED note style
        (
            f"EMERGENCY DEPARTMENT NOTE\n"
            f"Arrival: {visit_date}\n"
            f"Patient: {patient_name} | Age: {age} | MRN: {mrn}\n"
            f"DOB: {dob}\n\n"
            f"PRESENTING COMPLAINT: {cc.capitalize()}.\n\n"
            f"{patient_name} arrived via EMS to {hospital} on {visit_date}. "
            f"Age {age}. Vitals on arrival: BP 142/88, HR 96, RR 18, Temp 37.2C, SpO2 97%.\n\n"
            f"PMH: {dx}. Current medications include {med}.\n\n"
            f"DISPOSITION: Admitted for further workup.\n"
            f"Emergency contact phone: {phone}.\n"
            f"Address: {address}.\n"
            f"Attending: {doctor_name}\n"
        ),
    ]

    text = rng.choice(templates)

    # --- Locate all PHI spans ---
    phi_items = [
        (patient_name,   "PATIENT"),
        (doctor_name,    "STAFF"),
        (dob,            "DATE"),
        (visit_date,     "DATE"),
        (discharge_date, "DATE"),
        (phone,          "PHONE"),
        (address,        "LOC"),
        (age,            "AGE"),
        (mrn,            "ID"),
        (email,          "EMAIL"),
        (hospital,       "HOSP"),
    ]

    # Find ALL occurrences of each PHI value in the text
    spans = []
    for value, label in phi_items:
        search_start = 0
        while True:
            idx = text.find(value, search_start)
            if idx == -1:
                break
            spans.append({
                "start": idx,
                "end":   idx + len(value),
                "label": label,
                "text":  value,
            })
            search_start = idx + len(value)

    # Sort by start position and remove any accidental overlaps
    spans.sort(key=lambda s: s["start"])
    cleaned = []
    last_end = -1
    for span in spans:
        if span["start"] >= last_end:
            cleaned.append(span)
            last_end = span["end"]

    return text, cleaned


# ---------------------------------------------------------------------------
# Targeted note templates — emphasize specific PHI categories
#
# These templates place PHI in unusual or harder-to-detect contexts,
# giving the model more varied examples of its weak categories.
# ---------------------------------------------------------------------------

# Templates keyed by the PHI category they emphasize
TARGETED_TEMPLATES = {

    "AGE": [
        # Ages in narrative positions (harder to detect than structured "Age: XX")
        (
            "CLINICAL NOTE\n\n"
            "Attending: {doctor_name}\n"
            "Facility: {hospital}\n"
            "Date: {visit_date}\n\n"
            "{patient_name}, a {age}-year-old, was evaluated today. "
            "The patient is {age} years of age and has been experiencing symptoms "
            "for approximately two weeks. Given the patient's age of {age}, "
            "we recommend age-appropriate screening.\n\n"
            "MRN: {mrn} | DOB: {dob}\n"
            "Phone: {phone} | Email: {email}\n"
            "Address: {address}\n\n"
            "Plan: Follow up in 4 weeks.\n"
            "—{doctor_name}\n"
        ),
        (
            "CONSULT NOTE — {visit_date}\n\n"
            "Re: {patient_name} (MRN {mrn})\n"
            "DOB: {dob} (Age: {age})\n\n"
            "Thank you for this referral. I saw your {age}-year-old patient "
            "{patient_name} at {hospital} today. "
            "At age {age}, the differential diagnosis includes several "
            "age-related conditions.\n\n"
            "Contact: {phone}\n"
            "Address: {address}\n"
            "Attending: {doctor_name}\n"
        ),
    ],

    "LOC": [
        # Multiple addresses and location references
        (
            "PATIENT DEMOGRAPHICS UPDATE — {visit_date}\n\n"
            "Patient: {patient_name} (MRN: {mrn}, DOB: {dob})\n"
            "Age: {age}\n\n"
            "PRIMARY ADDRESS:\n{address}\n\n"
            "The patient relocated from {address} and now resides at "
            "the above address. Mail correspondence should be sent to "
            "{address}.\n\n"
            "Treatment facility: {hospital}\n"
            "Attending: {doctor_name}\n"
            "Contact: {phone} | {email}\n"
        ),
        (
            "HOME HEALTH REFERRAL\n\n"
            "Date: {visit_date}\n"
            "Patient: {patient_name} | Age: {age} | MRN: {mrn}\n"
            "DOB: {dob}\n\n"
            "Please arrange home health visits at the patient's residence: "
            "{address}. The patient was discharged from {hospital} on "
            "{discharge_date}. Nearest pharmacy is located near {address}.\n\n"
            "Physician: {doctor_name} ({phone})\n"
            "Patient email: {email}\n"
        ),
    ],

    "EMAIL": [
        # Emails in various contexts
        (
            "PATIENT COMMUNICATION LOG — {visit_date}\n\n"
            "Patient: {patient_name} (MRN: {mrn})\n"
            "DOB: {dob} | Age: {age}\n\n"
            "Email sent to {email} regarding lab results on {visit_date}. "
            "Patient responded via {email} confirming receipt. "
            "Please use {email} for all future correspondence. "
            "CC: {doctor_name} at {hospital}.\n\n"
            "Phone (backup): {phone}\n"
            "Address: {address}\n"
        ),
    ],

    "PHONE": [
        # Phone numbers in varied contexts
        (
            "TELEPHONE ENCOUNTER — {visit_date}\n\n"
            "Patient: {patient_name} (MRN: {mrn}, DOB: {dob})\n"
            "Age: {age}\n\n"
            "Called patient at {phone} to discuss test results. "
            "Patient answered at {phone}. Advised to call back at {phone} "
            "if symptoms worsen. Alternative contact: {email}.\n\n"
            "Patient address on file: {address}\n"
            "Facility: {hospital}\n"
            "Provider: {doctor_name}\n"
        ),
    ],

    "ID": [
        # MRN/ID in unusual positions
        (
            "RECORD MERGE NOTICE\n\n"
            "Date: {visit_date}\n"
            "Patient: {patient_name}\n"
            "DOB: {dob} | Age: {age}\n\n"
            "Medical record {mrn} has been identified as a duplicate. "
            "The primary record is {mrn}. All encounters under {mrn} "
            "have been consolidated at {hospital}.\n\n"
            "Attending: {doctor_name}\n"
            "Contact: {phone} | {email}\n"
            "Address: {address}\n"
        ),
    ],

    "PATIENT": [
        # Patient name repeated in many contexts
        (
            "MULTIDISCIPLINARY TEAM MEETING — {visit_date}\n\n"
            "Patient discussed: {patient_name} (MRN: {mrn})\n"
            "DOB: {dob} | Age: {age}\n\n"
            "{patient_name} was discussed at the weekly MDT meeting. "
            "The case of {patient_name} presents unique challenges. "
            "Dr. team agrees that {patient_name} would benefit from "
            "additional workup at {hospital}.\n\n"
            "Primary physician: {doctor_name}\n"
            "Phone: {phone} | Email: {email}\n"
            "Address: {address}\n"
        ),
    ],

    "STAFF": [
        # Doctor/staff names in varied positions
        (
            "SHIFT HANDOFF NOTE — {visit_date}\n\n"
            "From: {doctor_name}\n"
            "Facility: {hospital}\n\n"
            "Patient {patient_name} (MRN: {mrn}, DOB: {dob}, Age: {age}) "
            "was seen by {doctor_name} during the day shift. "
            "{doctor_name} initiated treatment and recommends continued "
            "monitoring. Please contact {doctor_name} at {phone} "
            "with any concerns.\n\n"
            "Patient contact: {email}\n"
            "Address: {address}\n"
        ),
    ],

    "DATE": [
        # Dates in many formats and contexts
        (
            "TIMELINE OF CARE\n\n"
            "Patient: {patient_name} (MRN: {mrn})\n"
            "DOB: {dob} | Age: {age}\n\n"
            "- {visit_date}: Initial presentation at {hospital}\n"
            "- {visit_date}: Labs drawn and imaging ordered\n"
            "- {discharge_date}: Discharge planned\n"
            "- {discharge_date}: Follow-up scheduled\n\n"
            "Attending: {doctor_name}\n"
            "Contact: {phone} | {email}\n"
            "Address: {address}\n"
        ),
    ],

    "HOSP": [
        # Hospital mentioned in many contexts
        (
            "TRANSFER NOTE\n\n"
            "Date: {visit_date}\n"
            "Patient: {patient_name} (MRN: {mrn}, DOB: {dob})\n"
            "Age: {age}\n\n"
            "Patient being transferred from {hospital} to {hospital} "
            "for specialized care. Originally admitted to {hospital} on "
            "{visit_date}. All records from {hospital} have been forwarded.\n\n"
            "Physician: {doctor_name} ({phone})\n"
            "Patient email: {email}\n"
            "Address: {address}\n"
        ),
    ],
}


def build_targeted_note(rng, target_category: str):
    """
    Build a synthetic note that emphasizes a specific PHI category.
    Uses targeted templates that place the category's PHI in varied,
    harder-to-detect contexts.

    Falls back to build_note() if no targeted template exists for the category.
    """
    templates = TARGETED_TEMPLATES.get(target_category)
    if not templates:
        return build_note(rng)

    # Generate PHI values
    patient_name   = random_name(rng)
    doctor_name    = random_doctor_name(rng)
    dob            = random_date(rng)
    visit_date     = random_date(rng)
    discharge_date = random_date(rng)
    phone          = random_phone(rng)
    address        = random_address(rng)
    age            = random_age(rng)
    mrn            = random_mrn(rng)
    email          = random_email(rng)
    hospital       = random_hospital(rng)

    template = rng.choice(templates)
    text = template.format(
        patient_name=patient_name,
        doctor_name=doctor_name,
        dob=dob,
        visit_date=visit_date,
        discharge_date=discharge_date,
        phone=phone,
        address=address,
        age=age,
        mrn=mrn,
        email=email,
        hospital=hospital,
    )

    # Locate all PHI spans (same logic as build_note)
    phi_items = [
        (patient_name,   "PATIENT"),
        (doctor_name,    "STAFF"),
        (dob,            "DATE"),
        (visit_date,     "DATE"),
        (discharge_date, "DATE"),
        (phone,          "PHONE"),
        (address,        "LOC"),
        (age,            "AGE"),
        (mrn,            "ID"),
        (email,          "EMAIL"),
        (hospital,       "HOSP"),
    ]

    spans = []
    for value, label in phi_items:
        search_start = 0
        while True:
            idx = text.find(value, search_start)
            if idx == -1:
                break
            spans.append({
                "start": idx,
                "end":   idx + len(value),
                "label": label,
                "text":  value,
            })
            search_start = idx + len(value)

    spans.sort(key=lambda s: s["start"])
    cleaned = []
    last_end = -1
    for span in spans:
        if span["start"] >= last_end:
            cleaned.append(span)
            last_end = span["end"]

    return text, cleaned


def generate_notes(
    num_notes: int,
    output_path: str,
    seed: int = 42,
    category_weights: dict = None,
) -> dict:
    """
    Generate synthetic notes and write to JSONL. Can be called programmatically
    by the loop orchestrator.

    Args:
        num_notes: Total number of notes to generate
        output_path: Path to write the JSONL file
        seed: Random seed
        category_weights: Optional dict of {category: weight} from error analysis.
            Higher weight = more targeted notes for that category.

    Returns:
        Dict with generation stats (label_counts, total_spans)
    """
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    label_counts = {}
    total_spans = 0

    # Decide how many notes go to each category vs. generic
    if category_weights:
        # Normalize weights to get proportions
        total_weight = sum(category_weights.values())
        cat_proportions = {
            cat: w / total_weight for cat, w in category_weights.items()
        }
        # Allocate notes: 40% generic, 60% targeted (distributed by weight)
        n_generic  = max(int(num_notes * 0.4), 1)
        n_targeted = num_notes - n_generic

        targeted_allocation = {}
        for cat, prop in cat_proportions.items():
            targeted_allocation[cat] = max(int(n_targeted * prop), 0)

        # Distribute any remainder
        allocated = sum(targeted_allocation.values())
        if allocated < n_targeted:
            # Give extras to the highest-weight category
            top_cat = max(category_weights, key=category_weights.get)
            targeted_allocation[top_cat] += n_targeted - allocated
    else:
        n_generic = num_notes
        targeted_allocation = {}

    with open(output_path, "w") as f:
        # Write generic notes
        for _ in range(n_generic):
            text, spans = build_note(rng)
            record = {"text": text, "spans": spans}
            f.write(json.dumps(record) + "\n")
            for span in spans:
                label_counts[span["label"]] = label_counts.get(span["label"], 0) + 1
                total_spans += 1

        # Write targeted notes
        for cat, count in targeted_allocation.items():
            for _ in range(count):
                text, spans = build_targeted_note(rng, cat)
                record = {"text": text, "spans": spans}
                f.write(json.dumps(record) + "\n")
                for span in spans:
                    label_counts[span["label"]] = label_counts.get(span["label"], 0) + 1
                    total_spans += 1

    actual_total = n_generic + sum(targeted_allocation.values())
    logger.info(f"Generated {actual_total} notes → {output_path}")
    if targeted_allocation:
        logger.info(f"  Generic: {n_generic} | Targeted: {sum(targeted_allocation.values())}")
        for cat, count in sorted(targeted_allocation.items(), key=lambda x: -x[1]):
            if count > 0:
                logger.info(f"    {cat:<12} {count:>4} notes")

    return {"label_counts": label_counts, "total_spans": total_spans, "num_notes": actual_total}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic PHI-annotated clinical notes")
    parser.add_argument("--num_notes", type=int, default=500,
                        help="Number of notes to generate (default: 500)")
    parser.add_argument("--output", type=str, default="data/synthetic_train.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--category_weights", type=str, default=None,
                        help='JSON string of category weights, e.g. \'{"AGE": 2.5, "LOC": 2.0}\'')
    args = parser.parse_args()

    # Parse category weights if provided
    weights = None
    if args.category_weights:
        try:
            weights = json.loads(args.category_weights)
            logger.info(f"Using category weights: {weights}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON for --category_weights: {args.category_weights}")
            raise

    stats = generate_notes(
        num_notes=args.num_notes,
        output_path=args.output,
        seed=args.seed,
        category_weights=weights,
    )

    print(f"\n✓ Generated {stats['num_notes']} notes → {args.output}")
    print(f"  Total PHI spans: {stats['total_spans']}")
    print(f"  Average spans per note: {stats['total_spans'] / max(stats['num_notes'], 1):.1f}")
    print(f"\n  Label distribution:")
    for label, count in sorted(stats["label_counts"].items(), key=lambda x: -x[1]):
        print(f"    {label:<12} {count:>5}  ({count/max(stats['total_spans'],1)*100:.1f}%)")
    print()


if __name__ == "__main__":
    main()

