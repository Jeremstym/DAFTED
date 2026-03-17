from enum import auto, unique
from typing import List, Tuple

from dataprocessing.data.config import LabelEnum
from strenum import SnakeCaseStrEnum, UppercaseStrEnum

PATIENT_ID_REGEX = r"\d{4}"
HDF5_FILENAME_PATTERN = "{patient_id}_{view}.h5"
IMG_FILENAME_PATTERN = "{patient_id}_{view}_{tag}{ext}"
IMG_FORMAT = ".nii.gz"
ATTRS_CACHE_FORMAT = "npz"
ATTRS_FILENAME_PATTERN = "{patient_id}{ext}"
TABULAR_ATTRS_FORMAT = "yaml"

IN_CHANNELS: int = 1
"""Number of input channels of the images in the dataset."""

DEFAULT_SIZE: Tuple[int, int] = (256, 256)
"""Default size at which the raw B-mode images are resized."""


@unique
class Label(LabelEnum):
    """Identifiers of the different anatomical structures available in the dataset's segmentation mask."""

    BG = 0
    """BackGround"""
    LV = 1
    """Left Ventricle"""
    MYO = 2
    """MYOcardium"""


@unique
class View(UppercaseStrEnum):
    """Names of the different views available for each patient."""

    A4C = auto()
    """Apical 4 Chamber"""
    A2C = auto()
    """Apical 2 Chamber"""
    A3C = auto()
    """Apical 3 Chamber"""


@unique
class TimeSeriesAttribute(SnakeCaseStrEnum):
    """Names of the attributes that are temporal sequences of values measured at each frame in the sequences."""

    gls = auto()
    ls_left = auto()
    ls_right = auto()
    """Longitudinal Strain (LS) of the endocardium: Global (GLS), left (LSL) and right (LSR)."""
    lv_area = auto()
    """Number of pixels covered by the left ventricle (LV)."""
    lv_length = auto()
    """Distance between the LV's base and apex."""
    myo_thickness_left = auto()
    myo_thickness_right = auto()
    """Average thickness of the myocardium (MYO) over left/right segments."""


@unique
class TabularAttribute(SnakeCaseStrEnum):
    """Name of the attributes that are scalar values extracted from the patient's record or the images."""

    ef = auto()
    """Ejection Fraction (EF)."""
    a2c_ef = "a2c_ef"
    """Ejection Fraction (EF) in the A2C view."""
    a4c_ef = "a4c_ef"
    """Ejection Fraction (EF) in the A4C view."""
    # a4c_edv = "a4c_edv"
    # """End-Diastolic Volume (EDV) in the A4C view."""
    # a2c_edv = "a2c_edv"
    # """End-Diastolic Volume (EDV) in the A2C view."""
    edv = auto()
    """End-Diastolic Volume (EDV)."""
    # a4c_esv = "a4c_esv"
    # """End-Systolic Volume (ESV) in the A4C view."""
    # a2c_esv = "a2c_esv"
    # """End-Systolic Volume (ESV) in the A2C view."""
    esv = auto()
    """End-Systolic Volume (ESV)."""
    a4c_ed_sc_min = "a4c_ed_sc_min"  # Assign the string manually because numbers are discarded by `auto`
    a4c_ed_sc_max = "a4c_ed_sc_max"  # ""
    a4c_ed_lc_min = "a4c_ed_lc_min"  # ""
    a4c_ed_lc_max = "a4c_ed_lc_max"  # ""
    a2c_ed_ic_min = "a2c_ed_ic_min"  # ""
    a2c_ed_ic_max = "a2c_ed_ic_max"  # ""
    a2c_ed_ac_min = "a2c_ed_ac_min"  # ""
    a2c_ed_ac_max = "a2c_ed_ac_max"  # ""
    a3c_ed_ilc_min = "a3c_ed_ilc_min"  # ""
    a3c_ed_ilc_max = "a3c_ed_ilc_max"  # ""
    a3c_ed_asc_min = "a3c_ed_asc_min"  # ""
    a3c_ed_asc_max = "a3c_ed_asc_max"  # ""
    """Peak concave(-) or convex(+) Curvatures of the Septal/Lateral Inferior/Anterior walls in A4C/A2C view, at ED."""
    age = auto()
    sex = auto()
    height = auto()
    weight = auto()
    bmi = auto()
    """Body Mass Index (BMI)."""
    stroke = auto()
    tobacco = auto()
    diabetes = auto()
    dyslipidemia = auto()
    oall = auto()
    tia = auto()
    ri = auto()
    ht = auto()
    fibrillation = auto()
    flutter = auto()
    valve_surgery = auto()
    valve_surgery_type = auto()
    iad = auto()
    cr_type = auto()
    """ Type of Coronary Revascularization (CR) performed on the patient."""
    pm = auto()
    """Whether the patient has a pacemaker (PM) or not."""
    cv_heredity = auto()
    """Whether the patient has a family history of cardiovascular disease."""
    diagnosis = auto()
    """Diagnosis of the patient."""
    # stratify = auto()
    # """Stratification for splitting the dataset."""
    arb = auto()
    """Whether the patient's treatment contains an Angiotensin Receptor Blocker (ARB) and which one."""
    tz_diuretic = auto()
    """Whether the patient's treatment contains a ThiaZide (TZ) diuretic and which one."""
    alpha_blocker = auto()
    """Whether the patient's treatment contains an alpha blocker and which one."""
    ccb = auto()
    """Whether the patient's treatment contains a Calcium Channel Blocker (CCB) and which one."""
    antipla = auto()
    anticoag = auto()
    cd_grade = auto()
    """Coronary Disease (CD) grade."""
    m_stenosis = auto()
    """Mitral Stenosis (MS) grade."""
    mi_grade = auto()
    """Mitral Insufficiency (MI) grade."""
    a_stenosis = auto()
    """Aortic Stenosis (AS) grade."""
    dilation = auto()
    """Aortic Dilation grade."""
    ti_grade = auto()
    """Tricuspid Insufficiency (TI) grade."""
    pi_grade = auto()
    """Pulmonary Insufficiency (PI) grade."""
    pericardium = auto()
    """Pericardium grade."""
    lp_diuritique = auto()
    k_diuritique = auto()
    arni = auto()
    sbp_tte = auto()
    """Systolic Blood Pressure (SBP) at the time of the TransThoracic Echocardiogram (TTE)."""
    dbp_tte = auto()
    """Diastolic Blood Pressure (SBP) at the time of the TransThoracic Echocardiogram (TTE)."""
    hr_tte = auto()
    """Heart Rate (HR) at the time of the TransThoracic Echocardiogram (TTE)."""
    creat = auto()
    """Serum creatinine concentrations."""
    gfr = auto()
    """Glomerular Filtration Rate (GFR)."""
    nt_probnp = auto()
    """NT-proBNP."""
    bnp = auto()
    hb = auto()
    """Hemoglobin (Hb) concentration."""
    proteinuria = auto()
    creatinuria = auto()
    microalbuminuria = auto()
    e_velocity = auto()
    """Peak E-wave (mitral passive inflow) velocity."""
    a_velocity = auto()
    """Peak A-wave (mitral inflow from active atrial contraction) velocity."""
    mv_dt = auto()
    """Mitral Valve (MV) Deceleration Time (DT)."""
    lateral_e_prime = auto()
    """Lateral mitral annular velocity."""
    septal_e_prime = auto()
    """Septal mitral annular velocity."""
    e_e_prime_ratio = auto()
    """Ratio of E over e'."""
    lvm = auto()
    """Left Ventricular Mass (LVM)."""
    lvm_ind = auto()
    """Left Ventricular Mass (LVM) index."""
    lvh = auto()
    """Whether the patient suffers from Left Ventricular Hypertrophy (LVH)."""
    ivs_d = auto()
    """InterVentricular Septum (IVS) thickness at end-diastole (D)."""
    lvid_d = auto()
    """Left Ventricular Internal Diameter (LVID) at end-diastole (D)."""
    pw_d = auto()
    """Left ventricular Posterior Wall (PW) thickness at end-diastole (D)."""
    tapse = auto()
    """Tricuspid Annular Plane Systolic Excursion (TAPSE)."""
    pisa_radius_mi = auto()
    sinus = auto()
    blood_flow = auto()
    a4c_lv_ev = "a4c_lv_ev"
    a2c_lv_ev = "a2c_lv_ev"
    vci_insp = auto()
    vci_exp = auto()
    la_dm = auto()
    la_area = auto()
    la_area_s = auto()
    # a4c_la_esv = "a4c_la_esv"
    # a2c_la_esv = "a2c_la_esv"
    la_esv = auto()
    ra_area_d = auto()
    pisa_vit_mi = auto()
    pisa_flow_mi = auto()
    roa_mi = auto()
    septal_e_e_prime_ratio = auto()
    lateral_e_e_prime_ratio = auto()
    sa_dm = auto()
    sa_area = auto()
    sa_vmax = auto()
    sa_vmean = auto()
    sa_pgmax = auto()
    sa_pgmean = auto()
    atv_vmax = auto()
    permeability_index = auto()
    atv_pgmax = auto()
    atv_vmean = auto()
    atv_pgmean = auto()
    atv_area = auto()
    sa_root_dm = auto()
    valsalva_sinus_dm = auto()
    st_junction_dm = auto()
    asc_aorta_dm = auto()
    pisa_radius_ti = auto()
    pisa_vit_ti = auto()
    roa_ti = auto()
    ti_pgmax = auto()
    arp = auto()
    paps = auto()
    pi_pgmax = auto()
    pi_vmax = auto()
    a4c_pld_isb = auto()
    a4c_pld_ism = auto()
    a4c_pld_sa = auto()
    a4c_pld_alm = auto()
    a4c_pld_alb = auto()
    a4c_pld_la = auto()
    a2c_pld_ib = auto()
    a2c_pld_im = auto()
    a2c_pld_ia = auto()
    a2c_pld_ab = auto()
    a2c_pld_aa = auto()
    a2c_pld_am = auto()
    gls = auto()
    sa_ev = auto()

    lbbb = auto()
    rbbb = auto()
    rhythm = auto()
    pr_interval = auto()
    qrs_interval = auto()
    qt_interval = auto()
    r_axis = auto()

    @classmethod
    def image_attrs(cls) -> List["TabularAttribute"]:
        """Lists the tabular attributes that are computed from the images."""
        return [
            TabularAttribute.ef,
            TabularAttribute.edv,
            TabularAttribute.esv,
            TabularAttribute.a4c_ed_sc_min,
            TabularAttribute.a4c_ed_sc_max,
            TabularAttribute.a4c_ed_lc_min,
            TabularAttribute.a4c_ed_lc_max,
            TabularAttribute.a2c_ed_ic_min,
            TabularAttribute.a2c_ed_ic_max,
            TabularAttribute.a2c_ed_ac_min,
            TabularAttribute.a2c_ed_ac_max,
        ]

    @classmethod
    def records_attrs(cls) -> List["TabularAttribute"]:
        """Lists the tabular attributes that come from the patient records."""
        return [attr for attr in cls if attr not in TabularAttribute.image_attrs()]

    @classmethod
    def categorical_attrs(cls) -> List["TabularAttribute"]:
        """Lists the tabular attributes that are categorical."""
        from dataprocessing.data.orchid.utils.attributes import TABULAR_CAT_ATTR_LABELS

        return [attr for attr in cls if attr in TABULAR_CAT_ATTR_LABELS]

    @classmethod
    def ordinal_attrs(cls) -> List["TabularAttribute"]:
        """Lists the subset of categorical attributes that are ordinal (i.e. the ordering of classes is meaningful)."""
        return [
            TabularAttribute.tobacco,
            TabularAttribute.cr_type,
            TabularAttribute.cd_grade,
            TabularAttribute.m_stenosis,
            TabularAttribute.mi_grade,
            TabularAttribute.a_stenosis,
            TabularAttribute.ti_grade,
            TabularAttribute.pi_grade,
            TabularAttribute.pericardium,
        ]

    @classmethod
    def binary_attrs(cls) -> List["TabularAttribute"]:
        """Lists the subset of categorical attributes that only have 2 classes (e.g. bool)."""
        from dataprocessing.data.orchid.utils.attributes import TABULAR_CAT_ATTR_LABELS

        return [attr for attr in cls.categorical_attrs() if len(TABULAR_CAT_ATTR_LABELS[attr]) == 2]

    @classmethod
    def boolean_attrs(cls) -> List["TabularAttribute"]:
        """Lists the subset of binary attributes that are boolean."""
        from dataprocessing.data.orchid.utils.attributes import TABULAR_CAT_ATTR_LABELS

        return [attr for attr in cls.binary_attrs() if TABULAR_CAT_ATTR_LABELS[attr] == [False, True]]

    @classmethod
    def numerical_attrs(cls) -> List["TabularAttribute"]:
        """Lists the tabular attributes that are numerical/continuous."""
        from dataprocessing.data.orchid.utils.attributes import TABULAR_CAT_ATTR_LABELS

        return [attr for attr in cls if attr not in TABULAR_CAT_ATTR_LABELS]

    @classmethod
    def tabular_attrs_shared(cls) -> List["TabularAttribute"]:
        """Lists the tabular attributes that are shared between the different views."""
        return [
            # TabularAttribute.diastolic_dysfunction_param_sum,
            TabularAttribute.lvm_ind,
            TabularAttribute.ivs_d,
            TabularAttribute.lvid_d,
            TabularAttribute.pw_d,
        ]

    @classmethod
    def tabular_attrs_unique(cls) -> List["TabularAttribute"]:
        """Lists the tabular attributes that are unique to each view."""
        return [attr for attr in cls if attr not in cls.tabular_attrs_shared()]


class OrchidTag(SnakeCaseStrEnum):
    """Tags referring to the different type of data stored."""

    # Tags describing data modalities
    time_series_attrs = auto()
    tabular_attrs = auto()

    # Tags referring to image data
    bmode = auto()
    mask = auto()
    resized_bmode = f"{bmode}_{DEFAULT_SIZE[0]}x{DEFAULT_SIZE[1]}"
    resized_mask = f"{mask}_{DEFAULT_SIZE[0]}x{DEFAULT_SIZE[1]}"

    # Tags for prefix/suffix to add for specific
    post = auto()

    # Tags referring to data attributes
    voxelspacing = auto()
