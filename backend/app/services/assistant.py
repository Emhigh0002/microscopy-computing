import re
from typing import Dict, List, Any

class AssistantService:
    def __init__(self):
        # Knowledge Base of Microorganisms
        self.knowledge_base = {
            "escherichia coli": {
                "name": "Escherichia coli (E. coli)",
                "category": "Gram-negative, rod-shaped bacterium",
                "staining": "Gram-negative (pink/red rod structures under Gram stain). Commonly stained using Crystal Violet, Gram's Iodine, Decolorizer, and Safranin counterstain.",
                "diseases": "Gastroenteritis, urinary tract infections (UTIs), neonatal meningitis, and hemolytic uremic syndrome (HUS).",
                "antibiotics": "Aminopenicillins (e.g., Ampicillin), Cephalosporins (e.g., Ceftriaxone), Fluoroquinolones (e.g., Ciprofloxacin), and Carbapenems (for ESBL strains).",
                "distinction": "Distinguished from other Enterobacteriaceae by its ability to ferment lactose (lactose positive on MacConkey agar) and positive indole test.",
                "sources": [
                    "CDC - Escherichia coli: CDC Center for Disease Control Guidelines (2024).",
                    "Bergey's Manual of Systematic Bacteriology, Vol 2: The Proteobacteria (2005).",
                    "Manual of Clinical Microbiology, 12th Edition (2019)."
                ]
            },
            "staphylococcus aureus": {
                "name": "Staphylococcus aureus (S. aureus)",
                "category": "Gram-positive, coccus-shaped (spherical) bacterium clustering in grape-like structures",
                "staining": "Gram-positive (violet/purple round clusters). Acid-fast negative.",
                "diseases": "Skin infections (impetigo, cellulitis), pneumonia, osteomyelitis, endocarditis, and toxic shock syndrome (TSS). Notable variant: MRSA.",
                "antibiotics": "Penicillinase-resistant penicillins (Nafcillin, Oxacillin), First-generation Cephalosporins, and Glycopeptides (Vancomycin for MRSA).",
                "distinction": "Catalase-positive and Coagulase-positive (which distinguishes it from Staphylococcus epidermidis and other coagulase-negative species).",
                "sources": [
                    "World Health Organization (WHO) - Guidelines on prevention of surgical site infections (2016).",
                    "ASM Microbe Library - Staphylococcus aureus characterization (2021).",
                    "Clinical Microbiology Reviews - Pathogenesis of Staphylococcus aureus infections (2015)."
                ]
            },
            "bacillus subtilis": {
                "name": "Bacillus subtilis (B. subtilis)",
                "category": "Gram-positive, endospore-forming, rod-shaped bacterium",
                "staining": "Gram-positive rod. Endospore staining (Schaeffer-Fulton method using Malachite Green and Safranin counterstain) reveals green oval endospores.",
                "diseases": "Generally considered non-pathogenic; can cause food contamination or opportunistic infections in immunocompromised hosts.",
                "antibiotics": "Broadly sensitive to beta-lactam antibiotics, Tetracyclines, and Chloramphenicol.",
                "distinction": "Obligate aerobe, motility positive, and produces highly resistant endospores positioned centrally or sub-terminally.",
                "sources": [
                    "Journal of Bacteriology - Biology of Bacillus subtilis Endospores (2008).",
                    "U.S. EPA - Bacillus subtilis Final Risk Assessment (1997)."
                ]
            },
            "candida albicans": {
                "name": "Candida albicans (C. albicans)",
                "category": "Diploid fungus (yeast)",
                "staining": "Stained using Gram stain (appears strongly Gram-positive/blue-purple ovals), KOH mount, or GMS (Gomori Methenamine Silver) stain.",
                "diseases": "Candidiasis (thrush, yeast infections), invasive systemic candidiasis in clinical settings.",
                "antibiotics": "Antifungals: Azoles (Fluconazole, Itraconazole), Polyenes (Amphotericin B, Nystatin), and Echinocandins.",
                "distinction": "Forms germ tubes when incubated in human serum at 37°C for 2-3 hours (Germ Tube Test positive), and forms chlamydoconidia on cornmeal agar.",
                "sources": [
                    "CDC - Fungal Diseases: Candidiasis (2023).",
                    "Principles of Internal Medicine - Harrison's, 21st Edition (2022)."
                ]
            },
            "spermatozoon": {
                "name": "Spermatozoon (Human Sperm Cell)",
                "category": "Male haploid gamete (specialized reproductive cell)",
                "staining": "Commonly stained using Papanicolaou (Papa) staining, Shorr stain, or rapid staining (e.g. Diff-Quik). Staining assists in morphological evaluations.",
                "diseases": "Associated clinical conditions: Teratozoospermia (abnormal morphology), Asthenozoospermia (poor motility), and Oligozoospermia (low count), contributing to male factor infertility.",
                "antibiotics": "Not applicable (non-microbial cell). However, in cases of bacteriospermia or leukocytospermia (semen infection), antibiotics like Doxycycline or Ciprofloxacin are indicated.",
                "distinction": "Characterized by three distinct structural zones: a flat oval head (3-5 µm length, containing the nucleus and acrosome), a midpiece (containing energy-producing mitochondria), and a long flagellar tail (approx. 45 µm long).",
                "sources": [
                    "WHO Laboratory Manual for the Examination and Processing of Human Semen, 6th Edition (2021).",
                    "European Society of Human Reproduction and Embryology (ESHRE) Guidelines (2020)."
                ]
            }
        }

    def answer_microbiology_question(self, query: str) -> Dict[str, Any]:
        normalized_query = query.lower()
        
        # Match organism keywords
        matched_key = None
        for key in self.knowledge_base:
            if key in normalized_query or (key == "spermatozoon" and "sperm" in normalized_query):
                matched_key = key
                break
                
        # Default response if no specific match
        if not matched_key:
            return {
                "response": "I am the Microscopy AI Assistant. I can answer questions about microorganism identification, Gram staining methods, antibiotic susceptibilities, and morphological traits. Please mention a specific organism such as Escherichia coli, Staphylococcus aureus, Bacillus subtilis, Candida albicans, or Spermatozoon to retrieve clinical specifications.",
                "citations": ["Standard Medical Microbiology Reference Library"]
            }
            
        data = self.knowledge_base[matched_key]
        
        # Determine focus of user query
        if any(w in normalized_query for w in ["stain", "dye", "color"]):
            ans = f"**Staining recommendation for {data['name']}:**\n{data['staining']}"
        elif any(w in normalized_query for w in ["disease", "pathology", "cause"]):
            ans = f"**Diseases associated with {data['name']}:**\n{data['diseases']}"
        elif any(w in normalized_query for w in ["antibiotic", "drug", "treat", "susceptibility"]):
            ans = f"**Recommended antibiotics / antimicrobial therapies for {data['name']}:**\n{data['antibiotics']}"
        elif any(w in normalized_query for w in ["distinguish", "character", "morphology", "identify"]):
            ans = f"**Morphological traits & diagnostic distinction of {data['name']}:**\n{data['distinction']}"
        else:
            # General summary response
            ans = (
                f"### {data['name']} Overview\n"
                f"* **Category**: {data['category']}\n"
                f"* **Gram Staining**: {data['staining']}\n"
                f"* **Associated Diseases**: {data['diseases']}\n"
                f"* **Typical Therapeutics**: {data['antibiotics']}\n"
                f"* **Diagnostic Distinction**: {data['distinction']}"
            )
            
        return {
            "response": ans,
            "citations": data["sources"]
        }

assistant_service = AssistantService()
