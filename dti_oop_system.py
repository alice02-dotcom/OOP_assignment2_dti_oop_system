from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Dict, List, Optional


# =========================
# Domain Classes
# =========================

@dataclass(frozen=True)
class Drug:
    """Represents one drug molecule."""
    drug_id: int
    cid: str
    canonical_smiles: str
    isomeric_smiles: str

    def short_smiles(self, max_len: int = 35) -> str:
        return self.canonical_smiles[:max_len] + "..." if len(self.canonical_smiles) > max_len else self.canonical_smiles


@dataclass(frozen=True)
class Protein:
    """Represents one target protein."""
    protein_id: int
    accession_number: str
    gene_name: str
    sequence: str

    def sequence_length(self) -> int:
        return len(self.sequence)


@dataclass(frozen=True)
class Interaction:
    """Represents a drug-target pair using composition."""
    drug: Drug
    protein: Protein

    @property
    def pair_id(self) -> str:
        return f"D{self.drug.drug_id}-P{self.protein.protein_id}"


@dataclass(frozen=True)
class PredictionResult:
    """Stores an interaction and its affinity score."""
    interaction: Interaction
    affinity_score: float

    def affinity_level(self) -> str:
        if self.affinity_score >= 8.0:
            return "High"
        elif self.affinity_score >= 6.0:
            return "Medium"
        return "Low"


# =========================
# Predictor Classes
# =========================

class AffinityPredictor(ABC):
    """
    Abstract parent class for affinity predictors.
    Different predictors can implement predict() in different ways.
    """

    @abstractmethod
    def predict(self, interaction: Interaction) -> PredictionResult:
        pass


class StoredAffinityPredictor(AffinityPredictor):
    """
    Uses already-existing affinity scores from the sample dataset.

    This project does not train or run a deep learning model.
    It focuses on object-oriented management of drug, protein,
    interaction, and prediction result objects.
    """

    def __init__(self, affinity_table: Dict[tuple[int, int], float]):
        self._affinity_table = affinity_table

    def predict(self, interaction: Interaction) -> PredictionResult:
        key = (interaction.drug.drug_id, interaction.protein.protein_id)

        if key not in self._affinity_table:
            raise ValueError(f"No affinity score exists for interaction {interaction.pair_id}")

        score = self._affinity_table[key]
        return PredictionResult(interaction=interaction, affinity_score=score)


class RuleBasedAffinityPredictor(AffinityPredictor):
    """
    A simple example predictor used only to demonstrate polymorphism.
    It does not represent a real biological prediction model.
    """

    def predict(self, interaction: Interaction) -> PredictionResult:
        smiles_length = len(interaction.drug.canonical_smiles)
        sequence_length = interaction.protein.sequence_length()

        # Simple artificial rule for demonstration only.
        score = 5.0 + ((smiles_length % 5) * 0.4) + ((sequence_length % 7) * 0.2)
        return PredictionResult(interaction=interaction, affinity_score=round(score, 3))


# =========================
# Data Loading Class
# =========================

class DataLoader:
    """Loads CSV files and converts rows into OOP objects."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def load_drugs(self, filename: str = "sample_drugs.csv") -> Dict[int, Drug]:
        drugs: Dict[int, Drug] = {}

        with open(self.data_dir / filename, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                drug = Drug(
                    drug_id=int(row["Drug_Index"]),
                    cid=str(row["CID"]),
                    canonical_smiles=row["Canonical_SMILES"],
                    isomeric_smiles=row["Isomeric_SMILES"],
                )
                drugs[drug.drug_id] = drug

        return drugs

    def load_proteins(self, filename: str = "sample_proteins.csv") -> Dict[int, Protein]:
        proteins: Dict[int, Protein] = {}

        with open(self.data_dir / filename, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                protein = Protein(
                    protein_id=int(row["Protein_Index"]),
                    accession_number=row["Accession_Number"],
                    gene_name=row["Gene_Name"],
                    sequence=row["Sequence"],
                )
                proteins[protein.protein_id] = protein

        return proteins

    def load_affinity_table(self, filename: str = "sample_interactions.csv") -> Dict[tuple[int, int], float]:
        affinity_table: Dict[tuple[int, int], float] = {}

        with open(self.data_dir / filename, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                drug_id = int(row["Drug_Index"])
                protein_id = int(row["Protein_Index"])
                affinity = float(row["Affinity"])
                affinity_table[(drug_id, protein_id)] = affinity

        return affinity_table

    def load_interactions(
        self,
        drugs: Dict[int, Drug],
        proteins: Dict[int, Protein],
        filename: str = "sample_interactions.csv",
    ) -> List[Interaction]:
        interactions: List[Interaction] = []

        with open(self.data_dir / filename, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                drug_id = int(row["Drug_Index"])
                protein_id = int(row["Protein_Index"])

                if drug_id not in drugs:
                    raise ValueError(f"Drug_Index {drug_id} does not exist in sample_drugs.csv")
                if protein_id not in proteins:
                    raise ValueError(f"Protein_Index {protein_id} does not exist in sample_proteins.csv")

                interactions.append(Interaction(drug=drugs[drug_id], protein=proteins[protein_id]))

        return interactions


# =========================
# Report Class
# =========================

class ReportManager:
    """Creates summaries from PredictionResult objects."""

    def __init__(self, results: List[PredictionResult]):
        self.results = results

    def average_affinity(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.affinity_score for result in self.results) / len(self.results)

    def best_result(self) -> Optional[PredictionResult]:
        if not self.results:
            return None
        return max(self.results, key=lambda result: result.affinity_score)

    def filter_by_level(self, level: str) -> List[PredictionResult]:
        return [result for result in self.results if result.affinity_level() == level]

    def print_summary(self) -> None:
        print("Drug-Target Interaction Analysis Summary")
        print("=" * 55)
        print(f"Number of results: {len(self.results)}")
        print(f"Average affinity: {self.average_affinity():.3f}")

        best = self.best_result()
        if best:
            print(
                "Best interaction: "
                f"{best.interaction.pair_id} "
                f"({best.interaction.protein.gene_name}) "
                f"Affinity={best.affinity_score:.3f}"
            )

        print("\nDetailed Results")
        print("-" * 55)
        for result in self.results:
            interaction = result.interaction
            print(
                f"{interaction.pair_id:8s} | "
                f"CID {interaction.drug.cid:8s} | "
                f"{interaction.protein.gene_name:10s} | "
                f"Affinity: {result.affinity_score:6.3f} | "
                f"Level: {result.affinity_level()}"
            )


# =========================
# Main Program
# =========================

def main() -> None:
    data_dir = Path(__file__).resolve().parent

    loader = DataLoader(data_dir)
    drugs = loader.load_drugs()
    proteins = loader.load_proteins()
    affinity_table = loader.load_affinity_table()
    interactions = loader.load_interactions(drugs, proteins)

    # Main predictor: uses affinity scores already included in the sample dataset.
    predictor: AffinityPredictor = StoredAffinityPredictor(affinity_table)
    results = [predictor.predict(interaction) for interaction in interactions]

    report = ReportManager(results)
    report.print_summary()

    # Polymorphism demonstration:
    # The same predict() call can use a different predictor object.
    print("\nPolymorphism Example : Using a simple artificial formula")
    print("-" * 55)
    demo_predictor: AffinityPredictor = RuleBasedAffinityPredictor()

    for interaction in interactions:
        demo_result = demo_predictor.predict(interaction)
        print(
            f"Using a simple artificial formula {demo_result.interaction.pair_id}: "
            f"Affinity={demo_result.affinity_score:.3f}, "
            f"Level={demo_result.affinity_level()}"
        )
    


if __name__ == "__main__":
    main()
