from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

class PropertyType(str, Enum):
    HOTEL = "Hotel"
    SCHOOL = "K-12 School"
    UNIVERSITY = "University"
    OFFICE_SMALL = "Small- and Mid-Sized Office"
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
    # Données catégorielles (via l'Enum)
    PrimaryPropertyType: PropertyType = Field(
        ..., 
        description="Type principal de bâtiment (doit être l'une des catégories connues)"
    )
    
    # Données numériques avec validations de base
    Latitude: float = Field(..., example=47.6062)
    Longitude: float = Field(..., example=-122.3321)
    YearBuilt: int = Field(..., ge=1900, le=2026, description="Année de construction")
    NumberofBuildings: int = Field(default=1, ge=0)
    NumberofFloors: int = Field(default=1, ge=0)
    PropertyGFATotal: float = Field(..., gt=0, description="Surface totale au sol")
    
    # Champ optionnel pour éviter que define_scope ne crash
    Outlier: Optional[str] = Field(default=None, description="Laisser vide")

    class Config:
        # Permet d'afficher un exemple complet dans Swagger
        json_schema_extra = {
            "example": {
                "PrimaryPropertyType": "Hotel",
                "Latitude": 47.612,
                "Longitude": -122.332,
                "YearBuilt": 1927,
                "NumberofBuildings": 1,
                "NumberofFloors": 11,
                "PropertyGFATotal": 200000.0
            }
        }