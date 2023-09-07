import eq
import pytorch_lightning as pl

# Load the earthquake catalog from (White et al. 2020)
catalog = eq.catalogs.White()

# Create train, validation and test dataloaders
dl_train = catalog.train.get_dataloader()
dl_val = catalog.val.get_dataloader()
dl_test = catalog.test.get_dataloader()

# Initialize and train the model
ntpp_model = eq.models.RecurrentTPP()
trainer = pl.Trainer(max_epochs=200)
trainer.fit(ntpp_model, dl_train)

# Compute negative log-likelihood on the test set
nll_test = trainer.test(ntpp_model, dl_test)
