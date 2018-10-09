
################ VERSIONS of packages used in this work
# Python 2.7.10
# keras.__version__ '1.1.0'
# numpy.__version__ '1.11.1'
# sklearn.__version__ '0.18'
# matplotlib.__version__ '1.5.3'
# argparse.__version__ '1.1'


import numpy as np
import argparse
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras.utils import np_utils
import keras.backend as K
from itertools import product
from functools import partial
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report


#Helper function: calculates f1 score for virus class only - we can track this value
def virus_F1(y_true,y_pred):
    virus = K.equal(K.argmax(y_true,axis=-1),K.ones_like(K.argmax(y_true,axis=-1)))
    predicted_virus = K.equal(K.argmax(y_pred,axis=-1),K.ones_like(K.argmax(y_true,axis=-1)))

    correct = K.equal(K.argmax(y_true,axis=-1), K.argmax(y_pred,axis=-1))
    TP = K.dot(virus,correct) #each element in dot wil be 1 iff "is virus" and "is true"

    virus_precision = K.sum(TP)/(K.sum(predicted_virus)+0.0001) #0.0001 is to avoid NaN
    virus_recall = K.sum(TP)/(K.sum(virus)+0.00001) #0.00001 is to avoid NaN
    F1 = 2*(virus_precision*virus_recall)/(virus_precision + virus_recall+0.0001) #0.0001 is to avoid NaN
    return F1 #virus_precision


#Helper function: defines any class weights we want (in a n x n matrix) - used this, beacuse class_weights did not seem to function
def weighted_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p] ,K.floatx())* K.cast(y_true[:, c_t],K.floatx()))
    return K.categorical_crossentropy(y_pred, y_true) * final_mask


#Helper function: get the project ID from the sequence ID (which is a string)
def seqID_to_projects(seqID):
    #projects = np.array([x[x.find("rr")+2:] for x in seqID])
    projects = np.array([x[x.find("_")+2:] for x in seqID])
    exp_names = np.unique(projects)
    print "EXPERIMENTS: ", exp_names, len(exp_names)
    pr_nr = np.zeros(np.shape(projects))
    for i,name in enumerate(exp_names):
        print i, name, np.shape(np.where(projects==name))
        pr_nr[np.where(projects==name)[0]] = i
    return pr_nr,exp_names


#------------------------------------
# this is the main object - MetaGenomics FeedForward Classifier
#------------------------------------
class Meta_FF_Classifier:
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

  #CREATING THE NETWORK
  def init(self, nb_inputs, nb_outputs):
    print "Creating model..."

    self.model = Sequential()

    #create n hidden layers
    for i in xrange(self.layers):
      if i == 0:
        layer = Dense(self.hidden_nodes, activation=self.activation, input_shape=(nb_inputs,))
      else:
	layer = Dense(self.hidden_nodes, activation=self.activation)
      self.model.add(layer)
      if self.dropout > 0: # add dropout after each layer
        self.model.add(Dropout(self.dropout))

    #and the final layer
    self.model.add(Dense(nb_outputs)) #no nonlinearity here before softmax
    self.model.add(Activation('softmax'))
    self.model.summary() #prints a summary

    if args.class_weight_power is not None:     #playing with class weights, using the custom loss function created above
      ncce = partial(weighted_categorical_crossentropy, weights=np.array([[args.class_weight_power[0],args.class_weight_power[0]],[args.class_weight_power[1],args.class_weight_power[1]]]))
      ncce.__name__ ='weighted_categorical_crossentropy'
      print "Compiling model with class weights..."
      self.model.compile(loss=ncce, optimizer=self.optimizer)
    else:  #if not using class weights
      print "Compiling model..."
      self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=[virus_F1])
  #end of defining the model


  #TRAINING the model
  def fit(self, train_X, train_y, valid_X, valid_y, save_path):
    #saving the model, normally (save_best_only=False) we do it after every epoch
    callbacks = [ModelCheckpoint(filepath=save_path, verbose=1, save_best_only=args.save_best_model_only)]  

    if self.lr_epochs > 0: #create LR scheduler if needed
      def lr_scheduler(epoch):
        lr = self.lr * self.lr_factor**int(epoch / self.lr_epochs)
        print "Epoch %d: learning rate %g" % (epoch + 1, lr)
        return lr
      callbacks.append(LearningRateScheduler(lr_scheduler))

    history = self.model.fit(train_X, train_y, batch_size=self.batch_size, nb_epoch=self.epochs, validation_data=(valid_X, valid_y), 
        shuffle=True, verbose=self.verbose, callbacks=callbacks)

  #EVALUATING the model
  def eval(self, X, y, load_path):
    self.model.load_weights(load_path) # this works even when evaluating during training

    pred_y = self.model.predict(X, batch_size=self.batch_size)
    pred_y_labels = np.argmax(pred_y, axis=1)

    report = classification_report(y, pred_y_labels, target_names=["not virus", "virus"])
    return report


  def load_and_pred(self, X, load_path):
    self.model.load_weights(load_path)
    pred_y = self.model.predict(X, batch_size=self.batch_size)
    print 'pred_y shape:', pred_y.shape
    return pred_y

  def predict(self, X):
    pred_y = self.model.predict(X, batch_size=self.batch_size)
    return pred_y

  #THRESHOLDING by output (does not prefectly reflect uncertainty)
  def plot_threshold(self, X, y, load_path):
    self.model.load_weights(load_path)
    pred_y = self.model.predict(X, batch_size=self.batch_size) #returns [P_non,P_vir]
    print 'plot threshold: pred_y shape ', pred_y.shape

    # we will create list with prediction and probability
    pred_and_prob = np.vstack([np.argmax(pred_y,axis=1),np.max(pred_y,axis=1)])
    #print np.min(np.max(pred_y,axis=1))
    pred_and_prob = np.transpose(pred_and_prob)

    pp=[]
    for threshold in np.arange(0.5,0.991,0.01): #for different thresholds
      pr_and_pr = pred_and_prob.copy()
      for i in range(pred_and_prob.shape[0]): #for all datapoints
        if pr_and_pr[i,1]<threshold: #if proba below threshold
          pr_and_pr[i,0] = -1 #set prediction to "no prediction"
      
      above = np.where(pr_and_pr[:,0]>-1)[0] #find the indexes of all instances above thresh
      trues = y[above]
      preds = pr_and_pr[above,0].flatten()

      pr=metrics.precision_score(trues,preds, average=None) #returns precision for each class (2 values)
      tr = (trues==preds)
      tp = trues*tr

      
      re=metrics.recall_score(y,pr_and_pr[:,0], average=None) #returns recall for each class (3 values, as -1 is class too)
      print "threshold", threshold,"prec", pr[1],"recall", re[-1]
      pp.append([pr[1],re[-1]]) #we take the prec and rec for only virus class

    pp= np.array(pp)
    plt.plot(pp[:,1]*100,pp[:,0]*100)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.axhline(90,color="r",linestyle="dashed")
    plt.savefig("precision_recall.png")
    plt.clf()
    plt.plot(np.arange(0.5,0.991,0.01),pp[:,0])
    plt.plot(np.arange(0.5,0.991,0.01),pp[:,1])
    plt.xlabel("threshold value")
    plt.ylabel("precision and recall")
    plt.savefig("threshold_precision_recall.png")
    plt.clf()


  def get_weights(self):
    return self.model.get_weights()

  def set_weights(self, weights):
    return self.model.set_weights(weights)


def str2bool(v): # a helper function that helps read binary parameter values from command line
  return v.lower() in ("yes", "true", "t", "1")

#
#---------------------------------------------------------------------------
#

# This part of the code is called only when you run "python network.py ....". 
# If you use an object of Meta_FF_Classifier class in another file this will not be called

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("save_path") # the most important arguments have no default value. program will crash if no value is given
  parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=2)

  parser.add_argument("--layers", type=int, choices=[1, 2, 3, 4, 5], default=2)
  parser.add_argument("--hidden_nodes", type=int, default=1024)
  parser.add_argument("--activation", choices=['tanh', 'relu'], default='relu')
  parser.add_argument("--dropout", type=float, default=0.25)
  parser.add_argument("--batch_size", type=int, default=10)
  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument("--train_shuffle", choices=['batch', 'true', 'false'], default='true')
  parser.add_argument("--optimizer", choices=['adam', 'rmsprop'], default='adam')
  parser.add_argument("--lr", type=float, default=0.001)
  parser.add_argument("--lr_epochs", type=int, default=1) # multiply LR with lr_factor after every lr_epochs
  parser.add_argument("--lr_factor", type=float, default=0.95)

  #the next 2 lines are different optinds to deal with unbalanced samples. Use only one.
  parser.add_argument("--class_weight_power", type=float, default=None)
  parser.add_argument("--oversample", type=float, default=None) # how many times we oversample the less populated class

  # True would mean we use validation set to decide when to stop training
  parser.add_argument("--save_best_model_only", type=str2bool, default=False)

  # we have loads of different datasets, so you need to pick one
  parser.add_argument("--datatype", choices=["concat","forward", "reverse"], default="concat")

  parser.add_argument("--leave_out", type=int, default=None) #allows to leave out experiments, based on 2ORF_projects_id_number.txt 
  parser.add_argument("--plot_only", type=str2bool, default=False) # use True for plotting pred,recall,F1 for already existing trained models

  args = parser.parse_args()
  print args._get_kwargs
  
  #args.save_path =  args.save_path+"{epoch:02d}"

  print "using datatype:", args.datatype
  if args.datatype == "concat": #DEFAULT#
	seqID1 = np.loadtxt("data/new_data/forward_al2/RSCU_id.txt",dtype=str)
        data1 = np.loadtxt("data/new_data/forward_al2/matrix.txt")
        values1 = np.loadtxt("data/new_data/forward_al2/values.txt",dtype=int)
        #projects1=seqID_to_projects(seqID1)
        
        #only_useful = np.where(np.sum(features1,axis=0)>np.array(np.sum(features1,axis=0),dtype=int))[0]
        data2 = np.loadtxt("data/new_data/reverse_al2/matrix.txt")
        values2 = np.loadtxt("data/new_data/reverse_al2/values.txt",dtype=int)
        seqID2 = np.loadtxt("data/new_data/reverse_al2/RSCU_id.txt",dtype=str)
        #pr2=seqID_to_projects(seqID2)

        data = np.concatenate([data1,data2])
        values = np.concatenate([values1,values2])
        seqID = np.concatenate([seqID1,seqID2])
 
        projects,exp_names=seqID_to_projects(seqID)    
  elif args.datatype == "forward":
     pass
  elif args.datatype == "reverse":
     pass       
  else: #return error if dataset name is not good
     assert(False)


  nb_options=2 # number of classes

  # when we leave projects out we need to cut away the data and leave this as validation set
  if (args.leave_out is not None) and (args.leave_out > -1):
    print "------------LEAVE OUT--------------"

    if args.leave_out > np.max(projects): #if exp nr out of bounds!!
      args.leave_out = None
      assert(False)
    lo_train_data=data[np.where(projects != args.leave_out)]
    lo_train_values=values[np.where(projects != args.leave_out)]
    lo_test_data=data[np.where(projects == args.leave_out)]
    lo_test_values=values[np.where(projects == args.leave_out)]   

  # usually we just split train and test randomly, but with leave-out we use the separation from above
  if args.leave_out is None:
    X_train, X_test, y_train, y_test = train_test_split(data, values, test_size=0.1, random_state=42)
  else:
    X_train = lo_train_data
    X_test= lo_test_data
    y_train = lo_train_values
    y_test = lo_test_values
    print "we are using EXPERIMENT LEAVE OUT (exp=",args.leave_out,"). \n Train and test sizes:", y_train.shape, y_test.shape  
  
  #print out statistics:
  counts = np.bincount(y_train)
  print "counts of classes in training set :", counts
  counts2 = np.bincount(y_test)
  print "counts of classes in test:", counts2, " virus shape ", np.shape(np.where(y_test==1))

  if len(counts2)==1:
	print "NO VIRUSES IN TEST SET, ABORTING"
	assert(False)

	
  # now dealing with oversampling 
  if args.oversample is not None:
    print "------------ PERFORMING OVERSAMPLING --------------"
    min_samples = np.min(counts)
    max_samples = np.max(counts)
    smaller_label = np.argmin(counts)
    bigger_label = np.argmax(counts)
    print "Oversampling - originally: smaller_label, min_samples", smaller_label, min_samples

    #we create a new matrix where we will add all the larger class and then the repetitions of smaller
    oversampled_train_X = X_train[np.where(y_train==bigger_label)[0],:]
    oversampled_train_Y = y_train[np.where(y_train==bigger_label)[0]]

    nr_sam = int(args.oversample * min_samples) #this is what we want in total
    n_repeat = np.repeat(np.where(y_train== smaller_label)[0], int(args.oversample)) #full repetitions of smaller class
    extra = np.random.choice(np.where(y_train==label)[0],nr_sam%counts[smaller_label]) #randomly picked extra samples
    smaller_samples = np.concatenate((n_repeat,extra))  # (this contains the indexes)
    oversampled_train_X = np.concatenate((oversampled_train_X, X_train[smaller_samples,:]), axis=0)
    oversampled_train_Y = np.concatenate((oversampled_train_Y, y_train[smaller_samples]), axis=0)

    #now we replace the training set with the oversampled training set    
    X_train = oversampled_train_X
    y_train = oversampled_train_Y

    print "After oversampling the shapes are", X_train.shape, y_train.shape
    counts = np.bincount(y_train)
    print "counts of classes after oversample:", counts
    counts2 = np.bincount(y_test)
    print "counts of classes in test after oversample (should stay the same):",counts2

  Y_train = np_utils.to_categorical(y_train, nb_options)  #turns it into one-hot vectors
  Y_test = np_utils.to_categorical(y_test, nb_options)  #turns it into one-hot vectors
  assert(np.all(y_train==np.argmax(Y_train,axis=1)))

  # if we use class weights, we want the weight to depend on
  # 1) nr of samples  and 2) a factor that allows us to make the weights more of less penalizing for bigger class
  # in total we multiply the loss with (max_count/count_of_this_class)**power. 
  # for bigger class its always 1. for smaller class it is >1, but can be bigger or smaller than (count_of_small/count_of_big) 
  if args.class_weight_power is not None:
    print "------------ CLASS WEIGHT CALCULATION --------------"
    counts = np.bincount(np.argmax(Y_train, axis=1))*1.0  # *1.0 to make it a float
    args.class_weight_power = (np.max(counts)/counts)**args.class_weight_power
    print "class weights", args.class_weight_power #from now on, cl_w_power contains the weights

  #Creating the model - an object of class Meta_FF_Classifier
  model = Meta_FF_Classifier(**vars(args))
  model.init(X_train.shape[1], Y_train.shape[1])

  #in case we only want to use a model, not train one
  if args.plot_only:
     model.plot_threshold(X_test, y_test, args.save_path+".hdf5")

  #when training
  else:
    model.fit(X_train, Y_train, X_test, Y_test, args.save_path + '.hdf5')
    print "----------------------train set---------------------------"
    report = model.eval(X_train, y_train, args.save_path + '.hdf5')
    print report
    print "----------------------test set---------------------------"
    report = model.eval(X_test, y_test, args.save_path + '.hdf5')
    print report
    #model.plot_threshold(X_test,y_test, args.save_path+".hdf5")
    predictions = model.predict(X_test)
    print "predictions shape",predictions.shape
    output = np.hstack((predictions, Y_test))
    np.savetxt("final_LOEO_cw/LOEO_predictions_"+str(args.leave_out)+".txt",output,fmt="%.3f")

