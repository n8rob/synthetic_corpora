There are two principle scripts:
	- trans.py
		This is for training translation systems.
		This script will both train the model and test it on 30% of the data that is available for testing. At the very end of execution, a BLEU score is printed out.
	- run_translation.py
		This is for translating with trained models.

Please see the "help" texts for the different arguments of these scripts in order to know how to run them.

The "second" bitext referenced in trans.py (associated with arguments src-data-second, tgt-data-second, src-second-name, and tgt-second), refers to whatever data should be kept out of the testing set. This would include synthetic bitexts or bitexts in any languages other than the objective language pair (i.e. the French-English bitext employed to train the bilingual model featured in the presentation).

Once you have a trained model, you may use run_translation.py for on-demand translation as demonstrated in the recorded demo. As shown in the demo, you need to pass the model name and the path to the model vocab file in order to translate. Both of these files will be saved automatically when you run trans.py.

The file run_translation.py can also be used to translate documents or set up human evaluations. See the mode argument in the file.

Here are the paths to all of the corpora used in the project thus far:

	- Authentic Haitian-English bitext:
		https://raw.githubusercontent.com/n8rob/corpora/master/ht_total.txt
		https://raw.githubusercontent.com/n8rob/corpora/master/en_total.txt		
	- Authentic French-English bitext:
		https://raw.githubusercontent.com/n8rob/corpora/master/enfr_fr_fromchurch.txt
		https://raw.githubusercontent.com/n8rob/corpora/master/enfr_en_fromchurch.txt
	- 'Synthetic Mono' synthetic Haitian-English bitext from back-translation 
		https://raw.githubusercontent.com/n8rob/synthetic_corpora/master/ht_enht_synth_95K.txt
		https://raw.githubusercontent.com/n8rob/synthetic_corpora/master/en_mono_95K.txt
	- 'Synthetic Mix 1' synthetic Haitian-English bitext
		https://raw.githubusercontent.com/n8rob/synthetic_corpora/master/ht_synth_mix1_115K.txt
		https://raw.githubusercontent.com/n8rob/synthetic_corpora/master/en_mono_mix1_115K.txt
	- 'Synthetic Mix 2' synthetic Haitian-English bitext
		https://raw.githubusercontent.com/n8rob/synthetic_corpora/master/ht_synth_mix2_125K_messup.txt
		https://raw.githubusercontent.com/n8rob/synthetic_corpora/master/en_mono_mix2_125K_messup.txt
	- Implicit French-Haitian bitext from two authentic bitexts:
		https://raw.githubusercontent.com/n8rob/synthetic_corpora/master/frht_french
		https://raw.githubusercontent.com/n8rob/synthetic_corpora/master/frht_haitian

Here are the commands to train the different models featured in the paper:

	- Unmodified
		python3 trans.py --src-data-first https://raw.githubusercontent.com/n8rob/corpora/master/ht_total.txt --tgt-data-first https://raw.githubusercontent.com/n8rob/corpora/master/en_total.txt --src-first-name enht_haitian --tgt-first-name enht_english --save-model-path ht2en_transln.pt

	- Bilingual
		python3 trans.py --src-data-first https://raw.githubusercontent.com/n8rob/corpora/master/ht_total.txt --tgt-data-first https://raw.githubusercontent.com/n8rob/corpora/master/en_total.txt --src-data-second https://raw.githubusercontent.com/n8rob/corpora/master/enfr_fr_fromchurch.txt --tgt-data-second https://raw.githubusercontent.com/n8rob/corpora/master/enfr_en_fromchurch.txt --src-first-name enht_haitian --tgt-first-name enht_english --src-second-name enfr_french --tgt-second-name enfr_english --save-model-path ht2en_biling_transln.pt

	- Synthetic Mono
		python3 trans.py --src-data-first https://raw.githubusercontent.com/n8rob/corpora/master/ht_total.txt --tgt-data-first https://raw.githubusercontent.com/n8rob/corpora/master/en_total.txt --src-data-second https://raw.githubusercontent.com/n8rob/synthetic_corpora/master/ht_enht_synth_95K.txt --tgt-data-second https://raw.githubusercontent.com/n8rob/synthetic_corpora/master/en_mono_95K.txt --src-first-name enht_haitian --tgt-first-name enht_english --src-second-name ht_enht_synth_95K.txt --tgt-second-name en_mono_95K.txt --save-model-path ht2en_augm_transln0.pt

	- Synthetic Mix 1
		python3 trans.py --src-data-first https://raw.githubusercontent.com/n8rob/corpora/master/ht_total.txt --tgt-data-first https://raw.githubusercontent.com/n8rob/corpora/master/en_total.txt --src-data-second https://raw.githubusercontent.com/n8rob/synthetic_corpora/master/ht_synth_mix1_115K.txt --tgt-data-second https://raw.githubusercontent.com/n8rob/synthetic_corpora/master/en_mono_mix1_115K.txt --src-first-name enht_haitian --tgt-first-name enht_english --src-second-name ht_synth_mix1_115K.txt --tgt-second-name en_mono_mix1_115K.txt --save-model-path ht2en_augm_transln1.pt
		
	- Synthetic Mix 2
		python3 trans.py --src-data-first https://raw.githubusercontent.com/n8rob/corpora/master/ht_total.txt --tgt-data-first https://raw.githubusercontent.com/n8rob/corpora/master/en_total.txt --src-data-second https://raw.githubusercontent.com/n8rob/synthetic_corpora/master/ht_synth_mix2_125K_messup.txt --tgt-data-second https://raw.githubusercontent.com/n8rob/synthetic_corpora/master/en_mono_mix2_125K_messup.txt --src-first-name enht_haitian --tgt-first-name enht_english --src-second-name ht_synth_mix2_125K.txt --tgt-second-name en_mono_mix2_125K.txt --save-model-path ht2en_augm_transln2.pt
