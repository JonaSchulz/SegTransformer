BaseTransformerBuilder:

| Parameter               | Data Type           | Explanation                                                                         |
|-------------------------|---------------------|-------------------------------------------------------------------------------------|
| n_layers                | int                 | The number of transformer layers                                                    |
| n_heads                 | int                 | The number of heads in each transformer layer                                       |
| feed_forward_dimensions | int                 | The dimensions of the fully connected layer in the transformer layers               |
| query_dimensions        | int                 | The dimensions of the queries and keys in each attention layer (per head)           |
| value_dimensions        | int                 | The dimensions of the values in each attention layer (per head)                     |
| dropout                 | float               | The dropout rate to be applied in the transformer encoder layer                     |
| activation              | str                 | The activation function for the transformer layer. One of {'relu', 'gelu'}          |
| final_normalization     | bool                | Whether to add LayerNorm as the final layer of the TransformerEncoder               |
| event_dispatcher        | str/EventDispatcher | The transformer event dispatcher either as a string or as an EventDispatcher object |

BaseTransformerEncoderBuilder:

| Parameter      | Data Type | Explanation                         |
|----------------|-----------|-------------------------------------|
| attention_type | str       | The attention implementation chosen |

BaseTransformerDecoderBuilder:

| Parameter            | Data Type | Explanation                                           |
|----------------------|-----------|-------------------------------------------------------|
| self_attention_type  | str       | The attention implementation used for self attention  |
| cross_attention_type | str       | The attention implementation used for cross attention |

