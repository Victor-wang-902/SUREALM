import torch.nn as nn
from transformers.models.bert.modeling_bert import BertSelfAttention, BertAttention, BertLayer, BertEncoder, BertModel

import math
import os
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.utils import (
    logging
)
from Embedding import ContextEmbed
import warnings
logger = logging.get_logger(__name__)

