devices = jax.devices()
if len(devices) > 1:
    raise ValueError("Farklı sayıda cihaza sahip olmak henüz desteklenmiyor.")
device = devices[0]
print("Çalıştırma Cihazı: ", device)


model_name_or_path = '' # --> Burada model adını veya yolu girin
model_type = '' # --> Burada model türünü giriniz (Seçenekler : 'gpt2' veya 'text-xxl')
num_tokens = None # --> Burada modelin kullanılabilir token sayısını girerek sınırlarını kontrol edin.
code_prompt = "" # --> Modelin kullandığı başlangıç kod parçası


openai.api_key = "" # --> Burada OpenAI API anahtarını girin 


train_state_config = {
    "learning_rate": 0.0001,
    "batch_size_per_device": 1,
    "num_train_steps": 1000, # --> Numara ayağa kalkacak iterasyonların sayısı
    "model_name_or_path": model_name_or_path,
    "model_type": model_type,
    "num_tokens": num_tokens,
    "code_prompt": code_prompt
}

train_state = train_state.TrainState.create(
    apply_fn=model_init,
    params=params,
    tx=tx,
    optimizer=optimizer,
    train=False,
    **train_state_config
)


dataset = load_dataset('') # --> Burada veri setini yüklemeye yardımcı olan bir fonksiyon kullanmanız gerekiyor. 
train_dataset = dataset['train']

def data_filter(data_item):
    # --> Verilerinizi filtreleyebilirsiniz (örn. Metni temizleyerek, zorluğu belirleyerek vb.)


context_length = 2048 # --> Burada Max TrainSequenceLength kullanılır. Bu sabit, "modelType" ile de ilgilidir.

batch_tokenizer = transformers.GPT2TokenizerFast.from_pretrained(model_name_or_path, pad_token="")

dataset = load_dataset("emotion")
 
def preprocess_function(examples):
    
    # Tokenize input texts to [CLS] + text + [SEP]
    result = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    
    label = examples['label']
    result['label'] = onehot(jnp.asarray(label), num_classes=8)
    
    return result
    
train_dataset = dataset['train'].map(preprocess_function, batched=True)
test_dataset = dataset['test'].map(preprocess_function, batched=True)

from transformers import FlaxAutoModel, AutoTokenizer

model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = FlaxAutoModel.from_pretrained(model_name, num_labels=8)

@dataclass
class ModelArguments:
    learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate."})
    num_train_epochs: int = field(default=3, metadata={"help": "Total number of training epochs to perform."})

args = ModelArguments()

def create_model():
    
    input_ids = jnp.ones((1, 128), dtype=jnp.int32)
    attention_mask = jnp.ones((1, 128), dtype=jnp.int32)
    encoder_output = base_model(input_ids=input_ids, attention_mask=attention_mask, deterministic=True).last_hidden_state
    
    logits = optax.Dense(num_classes)(encoder_output[:, 0])
    return logits

num_classes = 8
model_def = create_model()
params = model_def.init(jax.random.PRNGKey(seed=0))
opt = optax.chain(
        optax.adam(args.learning_rate),
        optax.clip_by_global_norm(1.0),
)

train_state = train_state.TrainState.create(
    apply_fn=create_model,
    params=params,
    tx=opt,
)

def cross_entropy_loss(logits, labels):
    logp = jax.nn.log_softmax(logits)
    loss = -jnp.sum(labels * logp)
    return loss / labels.shape[0]

def train_step(state, batch):
    inputs = shard(batch["input_ids"])
    attention_mask = shard(batch["attention_mask"])
    labels = shard(batch["label"])
    
    def loss_fn(params):
        logits = create_model().apply({'params': params}, inputs=inputs, attention_mask=attention_mask, deterministic=False)
        loss = cross_entropy_loss(logits, labels)
        return loss
    
    gradients = jax.grad(loss_fn)(state.params)
    gradients = jax.tree_map(lambda x: jax.lax.psum(x, axis=0), gradients)

    state = state.apply_gradients(gradients=gradients)
    return state


def eval_step(params, batch):
    inputs = shard(batch["input_ids"])
    attention_mask = shard(batch["attention_mask"])
    labels = shard(batch["label"])

    logits = create_model().apply({'params': params}, inputs=inputs, attention_mask=attention_mask, deterministic=True)
    loss = cross_entropy_loss(logits, labels)
    
    pred_probs = jax.nn.softmax(logits)
    preds = jnp.argmax(pred_probs, axis=-1)
    
    return {'loss': loss, 'accuracy': jnp.mean(preds == jnp.argmax(labels, axis=-1))}

for epoch in range(args.num_train_epochs):
    # Train
    for batch in train_dataset:
        train_state = train_step(train_state)
        
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
from datasets import load_dataset

dataset = load_dataset("text", data_files="path/to/your/data")
max_length = 512

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)

dataset = dataset.map(preprocess_function, batched=True)
# variables to set before training loop
learning_rate = 3e-4
weight_decay_rate = 0.01
batch_size = 4

num_train_epochs = 3
logging_steps = len(dataset) // batch_size

rng = jax.random.PRNGKey(0)

optimizer = optax.adamw(learning_rate=learning_rate, weight_decay_rate=weight_decay_rate)

train_state = train_state.TrainState.create(
    apply_fn=model.__call__,
    params=model.params,
    tx=optimizer
)


def compute_loss(logits, labels):
    return optax.softmax_cross_entropy(logits=logits, labels=onehot(labels, logits.shape[-1])).mean()


@jax.jit
def train_step(train_state, batch, rng):
    # prepare inputs and get the targets from the dataset
    inputs = shard(batch['input_ids'])
    targets = shard(batch['input_ids'])

    logits = train_state.apply_fn(input_ids=inputs, return_dict=True).logits

    loss = compute_loss(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)) 

    gradients = jax.grad(loss)(train_state.params) 
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
max_length = 512

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)

dataset = dataset.map(preprocess_function, batched=True)

learning_rate = 3e-4
weight_decay_rate = 0.01
batch_size = 4

num_train_epochs = 3
logging_steps = len(dataset) // batch_size

rng = jax.random.PRNGKey(0)

# define optimizer
optimizer = optax.adamw(learning_rate=learning_rate, weight_decay_rate=weight_decay_rate)

# build Flax train state
train_state = train_state.TrainState.create(
    apply_fn=model.__call__,
    params=model.params,
    tx=optimizer
)


def compute_loss(logits, labels):
    return optax.softmax_cross_entropy(logits=logits, labels=onehot(labels, logits.shape[-1])).mean()


@jax.jit
def train_step(train_state, batch, rng):
    # prepare inputs and get the targets from the dataset
    inputs = shard(batch['input_ids'])
    targets = shard(batch['input_ids'])

    logits = train_state.apply_fn(input_ids=inputs, return_dict=True).logits

    loss = compute_loss(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)) 

    gradients = jax.grad(loss)(train_state.params) 
