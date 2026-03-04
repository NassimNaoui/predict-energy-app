from pydantic import BaseModel, Field
from enum import Enum


class PropertyType(str, Enum):
    HOTEL = "Hotel"
    SCHOOL = "K-12 School"
    UNIVERSITY = "University"
    OFFICE_MID = "Small- and Mid-Sized Office"
    SELF_STORAGE = "Self-Storage Facility"
    WAREHOUSE = "Warehouse"
    OFFICE_LARGE = "Large Office"
    MEDICAL_OFFICE = "Medical Office"
    RETAIL_STORE = "Retail Store"
    HOSPITAL = "Hospital"
    DISTRIBUTION_CENTER = "Distribution Center"
    WORSHIP = "Worship Facility"
    SENIOR_CARE = "Senior Care Community"
    SUPERMARKET = "Supermarket / Grocery Store"
    LABORATORY = "Laboratory"
    REFRIGERATED_WAREHOUSE = "Refrigerated Warehouse"
    RESTAURANT = "Restaurant"


class EnergyInput(BaseModel):
    # --- Données Catégorielles ---
    PrimaryPropertyType: PropertyType = Field(
        ..., description="Type d'usage principal du bâtiment"
    )

    # --- Données Numériques ---
    PropertyGFATotal: float = Field(
        ..., gt=0, description="Surface totale du bâtiment (en pieds carrés)"
    )
    PropertyGFAParking: float = Field(
        default=0, ge=0, description="Surface dédiée au parking"
    )
    NumberofBuildings: int = Field(
        default=1, ge=1, description="Nombre de bâtiments sur la parcelle"
    )
    NumberofFloors: int = Field(default=1, ge=1, description="Nombre d'étages")
    YearBuilt: int = Field(
        ..., ge=1850, le=2026, description="Année de construction originelle"
    )

    class Config:
        # L'exemple qui apparaîtra directement dans l'interface Swagger
        json_schema_extra = {
            "example": {
                "PrimaryPropertyType": "Large Office",
                "PropertyGFATotal": 150000.0,
                "PropertyGFAParking": 25000.0,
                "NumberofBuildings": 1,
                "NumberofFloors": 12,
                "YearBuilt": 1985,
            }
        }
