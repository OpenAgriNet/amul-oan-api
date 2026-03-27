from pydantic import BaseModel, Field, field_validator, AliasChoices


class FarmerModel(BaseModel):
    state: str | None = None
    district: str | None = None
    sub_district: str | None = Field(None, alias="subDistrict")
    village: str | None = None
    union_name: str | None = Field(
        None, validation_alias=AliasChoices("unionName", "Union Name")
    )
    union_code: str | None = Field(None, alias="unionCode")
    society_name: str | None = Field(
        None, validation_alias=AliasChoices("societyName", "Society Name")
    )
    society_code: str | None = Field(None, alias="societyCode")
    farmer_name: str | None = Field(
        None, validation_alias=AliasChoices("farmerName", "Farm Name")
    )
    mobile_number: str | None = Field(
        None, validation_alias=AliasChoices("mobileNumber", "Mobile Number")
    )
    farmer_code: str | None = Field(None, alias="farmerCode")
    avg_milk_per_day_cow: float | None = Field(None, alias="avgMilkPerDayCow")
    avg_milk_per_day_buffalo: float | None = Field(None, alias="avgMilkPerDayBuff")
    cow_snf: float | None = Field(None, alias="cowSnf")
    cow_fat: float | None = Field(None, alias="cowFat")
    buff_snf: float | None = Field(None, alias="buffSnf")
    buff_fat: float | None = Field(None, alias="buffFat")
    animal_tags: list[str] | None = Field(None, alias="tagNo")
    total_animals: int | None = Field(
        None, validation_alias=AliasChoices("totalAnimals", "Total Animal")
    )
    total_cow: int | None = Field(None, validation_alias=AliasChoices("cow", "Cow"))
    total_buffalo: int | None = Field(
        None, validation_alias=AliasChoices("buffalo", "Buffalo")
    )
    total_milking_animals: int | None = Field(
        None, validation_alias=AliasChoices("totalMilkingAnimals", "Milking Animal")
    )
    non_pregnant_milking_animals: int | None = Field(None, alias="Non Pregnant Milk")
    pregnant_milking_animals: int | None = Field(None, alias="Pregnant Milk")

    @field_validator("animal_tags", mode="before")
    def transform_tagno(cls, tag_nos: str | None) -> list[str] | None:
        if tag_nos is None:
            return None
        return [tag_no.strip() for tag_no in tag_nos.strip().split(",")]

    @field_validator("union_name", mode="before")
    def transform_union_name(cls, union_name: str | None) -> str | None:
        if union_name is None:
            return None
        return union_name.strip().lower()

    @field_validator(
        "farmer_name",
        "state",
        "district",
        "sub_district",
        "village",
        "union_name",
        "society_name",
        mode="before",
    )
    def transform_pronouns(cls, pronoun: str | None) -> str | None:
        if pronoun is None:
            return None
        return pronoun.strip().lower()


class FarmerHerdmanModel(BaseModel):
    farmers: list[FarmerModel] | None = Field(None, alias="Farmer")
