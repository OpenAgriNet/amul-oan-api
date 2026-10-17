from pydantic import BaseModel, ConfigDict, Field, field_validator, AliasChoices

from agents.tools.models.local_names import prefer_local_name


class FarmerModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

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
    society_gujarati_name: str | None = Field(
        None,
        validation_alias=AliasChoices(
            "societyGujaratiName",
            "societyFullNamesGuj",
            "societyNameLocal",
        ),
    )
    society_code: str | None = Field(None, alias="societyCode")
    farmer_name: str | None = Field(
        None, validation_alias=AliasChoices("farmerName", "Farm Name")
    )
    farmer_gujarati_name: str | None = Field(
        None,
        validation_alias=AliasChoices(
            "farmerGujaratiName",
            "farmerFullNamesGuj",
            "farmerLocalName",
            "ownerFullNameInLocal",
        ),
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

    @property
    def display_farmer_name(self) -> str | None:
        """Farmer-facing name: Amul Gujarati when present, else English."""
        return prefer_local_name(self.farmer_gujarati_name, self.farmer_name)

    @property
    def display_society_name(self) -> str | None:
        """Society-facing name: Amul Gujarati when present, else English."""
        return prefer_local_name(self.society_gujarati_name, self.society_name)

    @field_validator("animal_tags", mode="before")
    def transform_tagno(cls, tag_nos: str | list[str] | None) -> list[str] | None:
        if tag_nos is None:
            return None
        if isinstance(tag_nos, list):
            return [tag_no.strip() for tag_no in tag_nos]
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

    @field_validator("farmer_gujarati_name", "society_gujarati_name", mode="before")
    @classmethod
    def strip_gujarati_names(cls, value: str | None) -> str | None:
        # Preserve Gujarati script/casing — never lowercase local-script names.
        if value is None:
            return None
        text = str(value).strip()
        return text or None
