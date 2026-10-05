from pydantic import BaseModel, Field


class CvccTreatmentMedicineModel(BaseModel):
    medicine_name: str | None = Field(None, alias="Medicine Name")
    medicine_dose: str | None = Field(None, alias="Medicine Dose")
    medicine_route: str | None = Field(None, alias="Medicine Route")


class CvccFodderDetailModel(BaseModel):
    fodder_group: str | None = Field(None, alias="Fodder Group")
    fodder_name: str | None = Field(None, alias="Fodder Name")
    fodder_qty_kg: str | None = Field(None, alias="Fodder QTY (Kg.)")


class CvccTreatmentModel(BaseModel):
    symptom: str | None = None
    treatment_date: str | None = Field(None, alias="Treatment Date")
    treatment: str | None = None
    medicine: list[CvccTreatmentMedicineModel] | None = None
    fodder_detail: list[CvccFodderDetailModel] | None = Field(
        None, alias="Fodder Detail"
    )


class CvccVaccinationModel(BaseModel):
    vaccine_name: str | None = Field(None, alias="vaccine Name")
    vaccination_type: str | None = Field(None, alias="vaccination Type")
    vaccination_date: str | None = Field(None, alias="vaccination Date")
    vaccine_for_disease: str | None = Field(None, alias="vaccine For Disease")


class CvccDewormingModel(BaseModel):
    deworming_date: str | None = Field(None, alias="Deworming Date")
    dewormer_name: str | None = Field(None, alias="Dewormer Name")
    dewormer_content: str | None = Field(None, alias="Dewormer Content")
    dewormer_dose: str | None = Field(None, alias="Dewormer Dose")


class CvccHealthDataModel(BaseModel):
    tag: str | None = Field(None, alias="Tag")
    animal_type: str | None = Field(None, alias="Animal Type")
    breed: str | None = None
    milking_stage: str | None = Field(None, alias="Milking Stage")
    pregnancy_stage: str | None = Field(None, alias="Pregnancy Stage ")
    lactation: str | None = Field(None, alias="Lactation")
    milk_yield: str | None = Field(None, alias="Milk Yield")
    farmer_mobile_number: str | None = Field(None, alias="Farmer mobile number")
    farmer_id: str | None = Field(None, alias="Farmer id")
    collar_belt: str | None = Field(None, alias="Coller Belt")
    treatment: list[CvccTreatmentModel] | None = Field(None, alias="Treatment")
    vaccination: list[CvccVaccinationModel] | None = Field(
        None, alias="Vaccination"
    )
    deworming: list[CvccDewormingModel] | None = Field(None, alias="Deworming")


class CvccHealthResponseModel(BaseModel):
    msg: str | None = None
    data: CvccHealthDataModel | None = None
