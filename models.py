from enum import Enum
from typing import Literal

from pydantic import BaseModel


class AggregationType(str, Enum):
    MIN = "Min"
    MAX = "Max"
    SUM = "Sum"
    AVG = "Avg"
    COUNT = "Count"
    COUNT_DISTINCT = "CountDistinct"
    MEDIAN = "Median"
    STDDEV = "StdDev"
    STDDEV_POP = "StdDevPop"
    VARIANCE = "Variance"
    VARIANCE_POP = "VariancePop"
    PREAGGREGATED = "preaggregated"


class TimeUnit(str, Enum):
    SECOND = "Second"
    MINUTE = "Minute"
    HOUR = "Hour"
    DAY = "Day"
    WEEK = "Week"
    MONTH = "Month"
    QUARTER = "Quarter"
    YEAR = "Year"


class SeriesType(str, Enum):
    BAR_GROUPED = "bar-grouped"
    BAR_STACKED = "bar-stacked"
    HISTOGRAM = "histogram"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    AREA_STACKED = "area-stacked"


## CORE OBJECTS (columns, basically)
class QualifiedTable(BaseModel):
    table: str
    schema: str
    database: str


class QualifiedColumn(BaseModel):
    # This is a "real column" from a table
    column: str
    table: str
    schema: str
    database: str


class ExploreCalculation(BaseModel):
    # This is a "fake column" on the table that is the result of an expression.
    # TODO: Actually represent this as an AST rather than a string.
    expression: str
    name: str


class TextDataColumn(BaseModel):
    source: QualifiedColumn | ExploreCalculation
    dtype: Literal["text"]


class NumericDataColumn(BaseModel):
    source: QualifiedColumn | ExploreCalculation
    dtype: Literal["numeric"]


class DateDataColumn(BaseModel):
    source: QualifiedColumn | ExploreCalculation
    dtype: Literal["date"]


class BooleanDataColumn(BaseModel):
    source: QualifiedColumn | ExploreCalculation
    dtype: Literal["boolean"]


# A DataColumn is basically either a base column or a calcuated column.
type DataColumn = TextDataColumn | NumericDataColumn | DateDataColumn | BooleanDataColumn


## PREDICATES and FILTERS


class UnaryTextPredicate(BaseModel):
    operator: Literal["IS_NULL", "IS_NOT_NULL"]


class UnaryNumericPredicate(BaseModel):
    operator: Literal["IS_NULL", "IS_NOT_NULL"]


class UnaryDatePredicate(BaseModel):
    operator: Literal["IS_NULL", "IS_NOT_NULL"]


class UnaryBooleanPredicate(BaseModel):
    operator: Literal["IS_NULL", "IS_NOT_NULL", "IS_TRUE", "IS_FALSE"]


class BinaryTextPredicate(BaseModel):
    operator: Literal["EQ", "NEQ", "CONTAINS", "DOES_NOT_CONTAIN"]
    text_value: str


class BinaryNumericPredicate(BaseModel):
    operator: Literal["EQ", "NEQ", "GT", "GTE", "LT", "LTE"]
    numeric_value: int | float


class ListTextPredicate(BaseModel):
    operator: Literal["IS_ONE_OF", "IS_NOT_ONE_OF"]
    list_values: list[str]


class ListNumericPredicate(BaseModel):
    operator: Literal["IS_ONE_OF", "IS_NOT_ONE_OF"]
    list_values: list[int | float]


class RelativeDatePredicate(BaseModel):
    operator: Literal["IS_AFTER", "IS_BEFORE", "IS_ON_OR_AFTER", "IS_ON_OR_BEFORE", "IS_ON", "IS_NOT_ON"]
    offset_value: int
    unit: TimeUnit
    direction: Literal["AGO", "FROM_NOW"]


class StaticDatePredicate(BaseModel):
    operator: Literal["IS_AFTER", "IS_BEFORE", "IS_ON_OR_AFTER", "IS_ON_OR_BEFORE", "IS_ON", "IS_NOT_ON"]
    date: str  # this is actually a date! TODO: make this a date object


# Conjunctive predicates represent the AND of all of the sub-predicates in their list.
class ConjunctiveTextPredicate(BaseModel):
    predicates: list[UnaryTextPredicate | BinaryTextPredicate | ListTextPredicate]


class ConjunctiveNumericPredicate(BaseModel):
    predicates: list[UnaryNumericPredicate | BinaryNumericPredicate | ListNumericPredicate]


class ConjunctiveDatePredicate(BaseModel):
    predicates: list[UnaryDatePredicate | RelativeDatePredicate | StaticDatePredicate]


class ConjunctiveBooleanPredicate(BaseModel):
    predicates: list[UnaryBooleanPredicate]


# Disjunctive predicates represent the OR of all of the sub-predicates in their list.
class DisjunctiveTextPredicate(BaseModel):
    predicates: list[UnaryTextPredicate | BinaryTextPredicate | ListTextPredicate]


class DisjunctiveNumericPredicate(BaseModel):
    predicates: list[UnaryNumericPredicate | BinaryNumericPredicate | ListNumericPredicate]


class DisjunctiveDatePredicate(BaseModel):
    predicates: list[UnaryDatePredicate | RelativeDatePredicate | StaticDatePredicate]


class DisjunctiveBooleanPredicate(BaseModel):
    predicates: list[UnaryBooleanPredicate]


type DisjunctivePredicate = (
    DisjunctiveTextPredicate | DisjunctiveNumericPredicate | DisjunctiveDatePredicate | DisjunctiveBooleanPredicate
)
type ConjunctivePredicate = (
    ConjunctiveTextPredicate | ConjunctiveNumericPredicate | ConjunctiveDatePredicate | ConjunctiveBooleanPredicate
)
type UnaryPredicate = UnaryTextPredicate | UnaryNumericPredicate | UnaryDatePredicate | UnaryBooleanPredicate
type BinaryPredicate = BinaryTextPredicate | BinaryNumericPredicate
type ListPredicate = ListTextPredicate | ListNumericPredicate
type DatePredicate = RelativeDatePredicate | StaticDatePredicate
type NumericPredicate = BinaryNumericPredicate | ListNumericPredicate
type TextPredicate = BinaryTextPredicate | ListTextPredicate
type BooleanPredicate = UnaryBooleanPredicate
type Predicate = (
    ConjunctivePredicate
    | DisjunctivePredicate
    | UnaryPredicate
    | BinaryPredicate
    | ListPredicate
    | DatePredicate
    | NumericPredicate
    | TextPredicate
    | BooleanPredicate
)


## CHARTS
class Series(BaseModel):
    series_type: SeriesType
    series_id: str


class PivotTableField(BaseModel):
    value: DataColumn
    channel: Literal["row", "column", "value"]
    series_id: str
    aggregation: AggregationType | None = None
    truncate_to_unit: TimeUnit | None = None


class ChartField(BaseModel):
    value: DataColumn
    channel: Literal["base-axis", "cross-axis", "color", "h-facet", "v-facet", "tooltip"]
    series_id: str
    aggregation: AggregationType | None = None
    truncate_to_unit: TimeUnit | None = None
    # TODO - do we need to make a ChartField for each of numeric/date/text/boolean?


class PivotTableConfig(BaseModel):
    fields: list[PivotTableField]
    series: list[Series]


class GroupedBarChartConfig(BaseModel):
    orientation: Literal["vertical", "horizontal"]
    fields: list[ChartField]
    series: list[Series]


class StackedBarChartConfig(BaseModel):
    orientation: Literal["vertical", "horizontal"]
    fields: list[ChartField]
    series: list[Series]


class HistogramConfig(BaseModel):
    fields: list[ChartField]
    series: list[Series]


class LineChartConfig(BaseModel):
    orientation: Literal["vertical", "horizontal"]
    fields: list[ChartField]
    series: list[Series]


class PieChartConfig(BaseModel):
    fields: list[ChartField]
    series: list[Series]


class ScatterPlotConfig(BaseModel):
    fields: list[ChartField]
    series: list[Series]


class AreaStackedConfig(BaseModel):
    fields: list[ChartField]
    series: list[Series]


class ExploreJoin(BaseModel):
    # Only support inner joins for now.
    source_table: QualifiedTable
    target_table: QualifiedTable
    source_column: QualifiedColumn
    target_column: QualifiedColumn


class ExploreFilter(BaseModel):
    data: DataColumn
    predicates: list[Predicate]


class ExploreSort(BaseModel):
    # We only support one sort column for now.
    data: DataColumn
    direction: Literal["asc", "desc"]


class ExploreStub(BaseModel):
    base_table: QualifiedTable
    joins: list[ExploreJoin] | None = None
    calculations: list[ExploreCalculation] = None
    filters: list[ExploreFilter] | None = None
    sort: ExploreSort | None = None
    visualization_config: (
        PivotTableConfig
        | GroupedBarChartConfig
        | StackedBarChartConfig
        | HistogramConfig
        | LineChartConfig
        | PieChartConfig
        | ScatterPlotConfig
        | AreaStackedConfig
    )
