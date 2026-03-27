import json
from typing import Any

from pydantic import BaseModel, Field, field_validator


class BanasMedicineModel(BaseModel):
    stock: int | float | None = None
    remarks: str | None = None
    uom_doctor: str | None = Field(None, alias="uomdoctor")
    uom_medicine: str | None = Field(None, alias="uommedicine")
    medicine_name: str | None = Field(None, alias="medicinename")


class BanasLabReportModel(BaseModel):
    sr_no: int | None = Field(None, alias="srno")
    remarks: str | None = None
    sample_date: str | None = Field(None, alias="sampledate")
    sample_name: str | None = Field(None, alias="samplename")


class BanasOperatedVisitModel(BaseModel):
    visit_code: str | None = Field(None, alias="VisitCode")
    visit_note_date: str | None = Field(None, alias="VisitNoteDate")
    visit_schedule_date: str | None = Field(None, alias="VisitScheduleDate")
    visit_allocation_date: str | None = Field(None, alias="VisitAllocationDate")
    visit_allocation_date_check: str | None = Field(
        None, alias="VisitAllocationDate_Check"
    )
    entry_time: str | None = Field(None, alias="EntryTime")
    visit_response_time: int | None = Field(None, alias="VisitResponseTime")
    species_name: str | None = Field(None, alias="speciesname")
    gender_name: str | None = Field(None, alias="gendername")
    pregnancy_status: str | None = Field(None, alias="pregnancystatus")
    breed: str | None = None
    milk_status: str | None = Field(None, alias="milkstatus")
    ailment_1: str | None = Field(None, alias="Ailment1")
    ailment_2: str | None = Field(None, alias="Ailment2")
    ailment_3: str | None = Field(None, alias="Ailment3")
    animal_tag_number: str | None = Field(None, alias="animaltagnumber")
    age_group: str | None = Field(None, alias="agegroup")
    society_name: str | None = Field(None, alias="societyname")
    society_code: str | None = Field(None, alias="societycode")
    society_phone_number: str | None = Field(None, alias="societyphonenumber")
    member_name: str | None = Field(None, alias="membername")
    member_code: str | None = Field(None, alias="membercode")
    member_address: str | None = Field(None, alias="memberaddress")
    member_contact_no: str | None = Field(None, alias="membercontactno")
    primary_doctor_name: str | None = Field(None, alias="primarydoctorname")
    secondary_doctor_name: str | None = Field(None, alias="secondarydoctorname")
    doctor_code: str | None = Field(None, alias="doctorcode")
    doctor_mobile: str | None = Field(None, alias="doctorMobile")
    driver_name: str | None = Field(None, alias="Drivername")
    disease: str | None = None
    disease_name: str | None = Field(None, alias="DiseaseName")
    disease_code_1: str | None = Field(None, alias="DiseaseCode1")
    medicine_remarks: str | None = Field(None, alias="medicineremarks")
    disease_group: str | None = Field(None, alias="diseasegroup")
    vehicle_reg_no: str | None = Field(None, alias="vehicleregno")
    payment_option: str | None = Field(None, alias="PaymentOption")
    payment_mode: str | None = Field(None, alias="PaymentMode")
    payment_comment: str | None = Field(None, alias="PaymentComment")
    vet_centre_name: str | None = Field(None, alias="VetcentreName")
    society_address: str | None = Field(None, alias="societyaddress")
    evp_details: str | None = Field(None, alias="evpdetails")
    prognosis_details: str | None = Field(None, alias="prognosisdetails")
    alloted_vet_centre_name: str | None = Field(None, alias="AllotedVetCentreName")
    medicines: list[BanasMedicineModel] | None = Field(None, alias="MedicinesJson")
    lab_reports: list[BanasLabReportModel] | None = Field(None, alias="LabReportsJson")
    report_date: str | None = Field(None, alias="ReportDate")
    visit_status: str | None = Field(None, alias="VisitStatus")

    @field_validator("medicines", mode="before")
    @classmethod
    def parse_medicines(cls, value: str | list[dict[str, Any]] | None) -> Any:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            return json.loads(value)
        return value

    @field_validator("lab_reports", mode="before")
    @classmethod
    def parse_lab_reports(cls, value: str | list[dict[str, Any]] | None) -> Any:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            return json.loads(value)
        return value
