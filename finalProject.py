import csv
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
# import keras
# import tensorflow
import pickle as pkl
import os
# import random
import time
# import sys
# import mingus
# import sklearn
from music21 import instrument, note, stream, chord, duration, converter
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.vis_utils import plot_model
from models.RNNAttention import get_distinct, create_lookups, prepare_sequences, \
    get_music_list, create_network, sample_with_temp
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def viewscore(filename, returnnotesdur=False, customfile=False, pitchaspitchspace=False):
    # filename = '24'
    if customfile:
        file = filename
    else:
        file = ".\\paganini\\capriccio{}.mid".format(filename)

    original_score = converter.parse(file).chordify()
    # original_score.show('text')

    notes = []
    durations = []

    for element in original_score.flat:
        if pitchaspitchspace:
            if isinstance(element, chord.Chord):
                curnotes = 1
                for n in element.pitches:
                    curnotes *= float(n.ps)
                notes.append(curnotes)
                durations.append(element.duration.quarterLength)

            if isinstance(element, note.Note):
                if element.isRest:
                    notes.append(0)
                    durations.append(element.duration.quarterLength)
                else:
                    notes.append(element.pitch.ps)
                    durations.append(element.duration.quarterLength)
        else:
            if isinstance(element, chord.Chord):
                notes.append('.'.join(n.nameWithOctave for n in element.pitches))
                durations.append(element.duration.quarterLength)

            if isinstance(element, note.Note):
                if element.isRest:
                    notes.append(str(element.name))
                    durations.append(element.duration.quarterLength)
                else:
                    notes.append(str(element.nameWithOctave))
                    durations.append(element.duration.quarterLength)

    """
    # print(f"# of notes:{len(notes)}")
    print('\nduration', 'pitch')
    for n, d in zip(notes, durations):
        print(d, '\t', n)
    """

    if returnnotesdur:
        return zip(notes, durations)
    else:
        return len(notes)


def getavgscorenoteduration():
    avgdurnotes = 0
    for i in range(1, 25):
        if i < 10:
            numstring = '0' + str(i)
        else:
            numstring = str(i)

        curnotes = viewscore(numstring)
        print(f"piece{i}: {curnotes} notes")
        avgdurnotes += curnotes

    avgdurnotes /= 24
    # print(f"average duration={avgdurnotes}")  # average=903     max=1415
    return avgdurnotes


def getmidiarrayinfo(midi_info, length, labeltoappend):
    arrmidi = []

    j = 0
    for n, d in midi_info:
        if j < length:
            arrmidi.append(float(n + d))
            j += 1
        else:
            arrmidi.append(labeltoappend)
            break

    return arrmidi


def trainmodel(run_id, music_name):
    # run params
    section = 'compose'
    # run_id = '0006'
    # music_name = 'paganini'

    run_folder = 'run\\{}\\'.format(section)
    run_folder += '_'.join([run_id, music_name])

    store_folder = os.path.join(run_folder, 'store')
    # data_folder = 'paganini'  # os.path.join('data', music_name)
    data_folder = music_name

    if not os.path.exists(run_folder):
        os.mkdir(run_folder)
        os.mkdir(os.path.join(run_folder, 'store'))
        os.mkdir(os.path.join(run_folder, 'output'))
        os.mkdir(os.path.join(run_folder, 'weights'))
        os.mkdir(os.path.join(run_folder, 'viz'))

    mode = 'build'  # 'load' #

    # data params
    intervals = range(1)
    seq_len = 32

    # model params
    embed_size = 100
    rnn_units = 256
    use_attention = True

    notes = []
    durations = []

    if mode == 'build':

        music_list, parser = get_music_list(data_folder)
        print(len(music_list), 'files in total')

        for i, file in enumerate(music_list):
            print(i + 1, "Parsing %s" % file)
            original_score = parser.parse(file).chordify()

            for interval in intervals:

                score = original_score.transpose(interval)

                notes.extend(['START'] * seq_len)
                durations.extend([0] * seq_len)

                for element in score.flat:

                    if isinstance(element, note.Note):
                        if element.isRest:
                            notes.append(str(element.name))
                            durations.append(element.duration.quarterLength)
                        else:
                            notes.append(str(element.nameWithOctave))
                            durations.append(element.duration.quarterLength)

                    if isinstance(element, chord.Chord):
                        notes.append('.'.join(n.nameWithOctave for n in element.pitches))
                        durations.append(element.duration.quarterLength)

        with open(os.path.join(store_folder, 'notes'), 'wb') as f:
            pkl.dump(notes, f)  # ['G2', 'D3', 'B3', 'A3', 'B3', 'D3', 'B3', 'D3', 'G2',...]
        with open(os.path.join(store_folder, 'durations'), 'wb') as f:
            pkl.dump(durations, f)
    else:
        with open(os.path.join(store_folder, 'notes'), 'rb') as f:
            notes = pkl.load(f)  # ['G2', 'D3', 'B3', 'A3', 'B3', 'D3', 'B3', 'D3', 'G2',...]
        with open(os.path.join(store_folder, 'durations'), 'rb') as f:
            durations = pkl.load(f)

    # get the distinct sets of notes and durations
    note_names, n_notes = get_distinct(notes)
    duration_names, n_durations = get_distinct(durations)
    distincts = [note_names, n_notes, duration_names, n_durations]

    with open(os.path.join(store_folder, 'distincts'), 'wb') as f:
        pkl.dump(distincts, f)

    # make the lookup dictionaries for notes and dictionaries and save
    note_to_int, int_to_note = create_lookups(note_names)
    duration_to_int, int_to_duration = create_lookups(duration_names)
    lookups = [note_to_int, int_to_note, duration_to_int, int_to_duration]

    with open(os.path.join(store_folder, 'lookups'), 'wb') as f:
        pkl.dump(lookups, f)

    # print(f'\nnote_to_int\n{note_to_int}')
    # print(f'\nduration_to_int\n{duration_to_int}')

    """ PREPARE NEURAL NETWORK """
    network_input, network_output = prepare_sequences(notes, durations, lookups, distincts, seq_len)

    print('pitch input')
    print(network_input[0][0])
    print('duration input')
    print(network_input[1][0])
    print('pitch output')
    print(network_output[0][0])
    print('duration output')
    print(network_output[1][0])

    model, att_model = create_network(n_notes, n_durations, embed_size, rnn_units, use_attention)
    model.summary()

    if not os.path.isfile('viz\\model.png'):  # keras plot_model library
        plot_model(model, to_file=os.path.join(run_folder, 'viz\\model.png'), show_shapes=True, show_layer_names=True)

    """ TRAIN NEURAL NETWORK """
    weights_folder = os.path.join(run_folder, 'weights')

    try:
        if os.path.isfile(os.path.join(weights_folder, "weights.h5")):
            model.load_weights(os.path.join(weights_folder, "weights.h5"))  # pickup where training left off
    except:
        print("Incompatible weights restored, generating new set")

    checkpoint1 = ModelCheckpoint(
        os.path.join(weights_folder, "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.h5"),
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )

    checkpoint2 = ModelCheckpoint(
        os.path.join(weights_folder, "weights.h5"),
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )

    early_stopping = EarlyStopping(
        monitor='loss',
        restore_best_weights=True,
        patience=10
    )

    callbacks_list = [
        checkpoint1,
        checkpoint2,
        early_stopping
    ]

    model.save_weights(os.path.join(weights_folder, "weights.h5"))
    model.fit(network_input, network_output,
              epochs=2000000, batch_size=32,
              validation_split=0.2,
              callbacks=callbacks_list,
              shuffle=True
              )


def predictmodel(run_id, music_name, max_seq_len=32, max_extra_notes=512):
    # run params
    section = 'compose'
    # run_id = '0006'
    # music_name = 'paganini'

    run_folder = 'run\\{}\\'.format(section)
    run_folder += '_'.join([run_id, music_name])

    # model params
    embed_size = 100
    rnn_units = 256
    use_attention = True

    store_folder = os.path.join(run_folder, 'store')

    """ GENERATE MODEL """

    with open(os.path.join(store_folder, 'distincts'), 'rb') as filepath:
        distincts = pkl.load(filepath)
        note_names, n_notes, duration_names, n_durations = distincts

    with open(os.path.join(store_folder, 'lookups'), 'rb') as filepath:
        lookups = pkl.load(filepath)
        note_to_int, int_to_note, duration_to_int, int_to_duration = lookups

    weights_folder = os.path.join(run_folder, 'weights')
    weights_file = 'weights.h5'

    model, att_model = create_network(n_notes, n_durations, embed_size, rnn_units, use_attention)

    # Load the weights to each node
    weight_source = os.path.join(weights_folder, weights_file)
    model.load_weights(weight_source)
    model.summary()

    """ BUILD PREDICTION """

    # prediction params
    notes_temp = 0.5
    duration_temp = 0.5
    # max_extra_notes = 512  # default=50 500
    # max_seq_len = 32  # int(random.randint(32, 128) / 32) * 32  # default=32 max 135 measures 903
    seq_len = 32  # default=32

    # notes = ['START', 'D3', 'D3', 'E3', 'D3', 'G3', 'F#3','D3',
    #   'D3', 'E3', 'D3', 'G3', 'F#3','D3', 'D3', 'E3', 'D3', 'G3', 'F#3','D3', 'D3', 'E3', 'D3', 'G3', 'F#3']
    # durations = [0, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2]

    # notes = ['START', 'F#3', 'G#3', 'F#3', 'E3', 'F#3', 'G#3', 'F#3', 'E3', 'F#3',
    #   'G#3', 'F#3', 'E3','F#3', 'G#3', 'F#3', 'E3', 'F#3', 'G#3', 'F#3', 'E3', 'F#3', 'G#3', 'F#3', 'E3']
    # durations = [0, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2]

    notes = ['START']
    durations = [0]

    if seq_len is not None:
        notes = ['START'] * (seq_len - len(notes)) + notes
        durations = [0] * (seq_len - len(durations)) + durations

    sequence_length = len(notes)

    """ GENERATE NN PREDICTION """

    prediction_output = []
    notes_input_sequence = []
    durations_input_sequence = []

    overall_preds = []

    for n, d in zip(notes, durations):
        note_int = note_to_int[n]
        duration_int = duration_to_int[d]

        notes_input_sequence.append(note_int)
        durations_input_sequence.append(duration_int)

        prediction_output.append([n, d])

        if n != 'START':
            midi_note = note.Note(n)

            new_note = np.zeros(128)
            new_note[midi_note.pitch.midi] = 1
            overall_preds.append(new_note)

    att_matrix = np.zeros(shape=(max_extra_notes + sequence_length, max_extra_notes))

    for note_index in range(max_extra_notes):

        prediction_input = [
            np.array([notes_input_sequence]),
            np.array([durations_input_sequence])
        ]

        notes_prediction, durations_prediction = model.predict(prediction_input, verbose=0)
        if use_attention:
            att_prediction = att_model.predict(prediction_input, verbose=0)[0]
            att_matrix[(note_index - len(att_prediction) + sequence_length):
                       (note_index + sequence_length), note_index] = att_prediction

        new_note = np.zeros(128)

        for idx, n_i in enumerate(notes_prediction[0]):
            try:
                note_name = int_to_note[idx]
                midi_note = note.Note(note_name)
                new_note[midi_note.pitch.midi] = n_i
            except:
                pass

        overall_preds.append(new_note)

        i1 = sample_with_temp(notes_prediction[0], notes_temp)
        i2 = sample_with_temp(durations_prediction[0], duration_temp)

        note_result = int_to_note[i1]
        duration_result = int_to_duration[i2]

        prediction_output.append([note_result, duration_result])

        notes_input_sequence.append(i1)
        durations_input_sequence.append(i2)

        if len(notes_input_sequence) > max_seq_len:
            notes_input_sequence = notes_input_sequence[1:]
            durations_input_sequence = durations_input_sequence[1:]

        # print(note_result)
        # print(duration_result)

        if note_result == 'START':
            break

    overall_preds = np.transpose(np.array(overall_preds))
    print('Generated sequence of {} notes'.format(len(prediction_output)))

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_yticks([int(j) for j in range(35, 70)])

    plt.imshow(overall_preds[35:70, :], origin="lower", cmap='coolwarm', vmin=-0.5, vmax=0.5,
               extent=[0, max_extra_notes, 35, 70]
               )

    fig.savefig('finalProject_sequence_plot.png')

    """ GENERATE MIDI """
    output_folder = os.path.join(run_folder, 'output')

    midi_stream = stream.Stream()

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        note_pattern, duration_pattern = pattern
        # pattern is a chord
        if '.' in note_pattern:
            notes_in_chord = note_pattern.split('.')
            chord_notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(current_note)
                new_note.duration = duration.Duration(duration_pattern)
                new_note.storedInstrument = instrument.Violin()
                chord_notes.append(new_note)
            new_chord = chord.Chord(chord_notes)
            midi_stream.append(new_chord)
        elif note_pattern == 'rest':
            # pattern is a rest
            new_note = note.Rest()
            new_note.duration = duration.Duration(duration_pattern)
            new_note.storedInstrument = instrument.Violin()
            midi_stream.append(new_note)
        elif note_pattern != 'START':
            # pattern is a note
            new_note = note.Note(note_pattern)
            new_note.duration = duration.Duration(duration_pattern)
            new_note.storedInstrument = instrument.Violin()
            midi_stream.append(new_note)

    midi_stream = midi_stream.chordify()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    midi_stream.write('midi', fp=os.path.join(output_folder, 'output-' + timestr + '.mid'))

    """ GENERATE ATTENTION PLOT """
    if use_attention:
        fig, ax = plt.subplots(figsize=(20, 20))

        # im = ax.imshow(att_matrix[(seq_len - 2):, ], cmap='coolwarm', interpolation='nearest')

        # Minor ticks
        ax.set_xticks(np.arange(-.5, len(prediction_output) - seq_len, 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(prediction_output) - seq_len, 1), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(prediction_output) - seq_len))
        ax.set_yticks(np.arange(len(prediction_output) - seq_len + 2))
        # ... and label them with the respective list entries
        ax.set_xticklabels([n[0] for n in prediction_output[seq_len:]])
        ax.set_yticklabels([n[0] for n in prediction_output[(seq_len - 2):]])

        # ax.grid(color='black', linestyle='-', linewidth=1)

        ax.xaxis.tick_top()

        plt.setp(ax.get_xticklabels(), rotation=90, ha="left", va="center",
                 rotation_mode="anchor")

        plt.show()
        fig.savefig('finalProject_attention_plot.png')


def svmdiscriminator(generated_midi_path, csvname, numdims, verbose=False, nondefpath=False, pathtocheck=""):
    csvpath = 'data\\{}.csv'.format(csvname)
    if os.path.isfile(csvpath):
        # print("File exist")
        data = np.genfromtxt(csvpath, delimiter=',')
    else:
        # print("File not exist")
        data = np.empty((0, numdims + 1))  # 431

        if not nondefpath:
            print("SVM converting pieces")
            for i in range(1, 25):
                # print(f"SVM converting caprice #{i}")
                if i < 10:
                    numstring = '0' + str(i)
                else:
                    numstring = str(i)

                midi_info = viewscore(numstring, True, pitchaspitchspace=True)
                arrmidi = getmidiarrayinfo(midi_info, numdims, 0)  # 430

                # print(arrmidi)
                # print(f"\tlen: {len(arrmidi)}")
                data = np.vstack([data, arrmidi])
        else:
            print("SVM converting pieces")
            music_list, parser = get_music_list(pathtocheck)
            for i, file in enumerate(music_list):
                # print(i + 1, "Parsing %s" % file)
                midi_info = viewscore(file, True, True, True)
                arrmidi = getmidiarrayinfo(midi_info, numdims, 0)  # 430

                # print(arrmidi)
                # print(f"\tlen: {len(arrmidi)}")
                data = np.vstack([data, arrmidi])

        with open("data\\{}.csv".format(csvname), "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        # print(allmidis)

    # print(f"SVM converting ai composition")
    midi_info = viewscore(generated_midi_path, True, True, True)
    arrmidi = getmidiarrayinfo(midi_info, numdims, 1)  # 430

    # print(arrmidi)
    # print(f"\tlen: {len(arrmidi)}")
    data = np.vstack([data, arrmidi])
    # print(f"shape={data.shape}")
    # print(data)
    # print("***Now performing SVM***")

    n = 25
    if nondefpath:
        music_list, parser = get_music_list(pathtocheck)
        n = len(music_list) + 1
        # print(n)

    d = numdims  # number of dimensions (default=430)
    label = data[:, d]
    # print(label)
    data = data[:, 0:d]

    while True:
        try:
            p = np.random.permutation(n)  # permutations

            training_size = int(round(.7 * n))
            training_data = data[p[0:training_size], :]
            training_label = label[p[0:training_size]]
            # print(training_label)

            test_data = data[p[training_size:n], :]
            test_label = label[p[training_size:n]]

            # classifier = SVC(C=.001, kernel='rbf')  # C values must be on a magnitude of 10 (0.1, .1, 1, 10, 100, ...)
            classifier = SVC(C=.001, kernel='linear')
            # classifier = SVC(C=.1, kernel='sigmoid')
            # classifier = SVC(C=.000001, kernel='poly', degree=2)  # for poly, keep C low or runtime slow

            classifier.fit(training_data, training_label)  # create w and b that will best fit data and label
            test_predict = classifier.predict(test_data)

            accuracy = accuracy_score(test_label, test_predict)
            if verbose:
                print(f"accuracy={round(accuracy * 100, 3)}%")

            return accuracy
        except:
            # print("Error caught due to permutations, retrying...")
            continue


def averagesvm(generated_midi_path, num_iterations, csvname, numdims, verbose=False, nondefpath=False, pathtocheck=""):
    averageacc = 0
    for i in range(0, num_iterations):
        averageacc += svmdiscriminator(generated_midi_path, csvname, numdims, verbose, nondefpath, pathtocheck)

    averageacc /= num_iterations
    print(generated_midi_path)
    print(f"Average accuracy across {num_iterations} iterations={round(averageacc * 100, 3)}%")
    return averageacc


def fpmain():
    # viewscore('24')
    # trainmodel('0003', 'moderato')
    # for i in range(0, 30):
    #     predictmodel('0003', 'moderato')
    # for i in range(0, 30):
    #     predictmodel('0003', 'moderato', max_seq_len=int(random.randint(32, 128) / 32) * 32)

    # trainmodel('0003', 'presto')
    # for i in range(0, 30):
    #     predictmodel('0003', 'presto')
    # for i in range(0, 30):
    #     predictmodel('0003', 'presto', max_seq_len=int(random.randint(32, 128) / 32) * 32)

    # getavgscorenoteduration()

    averagesvm('run/compose/0003_presto/output/--------output-20201130-180054.mid',
               100, 'presto', 430, verbose=False, nondefpath=True, pathtocheck="presto")
    averagesvm('run/compose/0003_presto/output/--------output-20201130-185416.mid',
               100, 'presto', 430, verbose=False, nondefpath=True, pathtocheck="presto")
    averagesvm('run/compose/0003_presto/output/--------output-20201130-200651.mid',
               100, 'presto', 430, verbose=False, nondefpath=True, pathtocheck="presto")
    print("\n\n")
    averagesvm('run/compose/0003_presto/output/--------output-20201130-180054.mid',
               100, 'paganini', 430, verbose=False)
    averagesvm('run/compose/0003_presto/output/--------output-20201130-185416.mid',
               100, 'paganini', 430, verbose=False)
    averagesvm('run/compose/0003_presto/output/--------output-20201130-200651.mid',
               100, 'paganini', 430, verbose=False)
    # Note to self: if training on shorter sequences fails,
    #   try smaller sequences and perform prediction on existing snippets

    
fpmain()
