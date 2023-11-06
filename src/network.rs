use std::{
    fs::File,
    io::Read,
    path::Path
};

use serde::{Serialize, Deserialize};

pub use containers::{LayerType, LayerInnerType};

use optimizers::*;

mod optimizers;

mod containers;


pub type CurrentOptimizer = Adam;

pub trait NewableLayer
{
    fn new(previous_size: usize, this_size: usize) -> Self;
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ThisWeightsContainer<T>
{
    weights: Vec<T>,
    biases: Vec<T>
}

impl ThisWeightsContainer<LayerType>
{
    pub fn new_randomized(layout: &[usize]) -> Self
    {
        let biases = layout.iter().copied().skip(1).map(|current_size|
        {
            let layer = LayerInnerType::new(current_size, 1);

            LayerType::new_diff(layer)
        }).collect();

        let weights = layout.iter().copied().zip(layout.iter().skip(1).copied())
            .map(|(previous_size, current_size)|
            {
                let layer = LayerInnerType::new_with(current_size, previous_size, ||
                {
                    let v = 1.0 / (previous_size as f32).sqrt();

                    (fastrand::f32() * 2.0 - 1.0) * v
                });

                LayerType::new_diff(layer)
            }).collect();

        Self{weights, biases}
    }
}

impl<T: NewableLayer> ThisWeightsContainer<T>
{
    pub fn new_container(layout: &[usize]) -> Self
    {
        let biases = layout.iter().copied().skip(1).map(|current_size|
        {
            T::new(current_size, 1)
        }).collect();

        let weights = layout.iter().copied().zip(layout.iter().skip(1).copied())
            .map(|(previous_size, current_size)|
            {
                T::new(current_size, previous_size)
            }).collect();

        Self{weights, biases}
    }
}

impl<T> ThisWeightsContainer<T>
{
    pub fn weights_biases_mut(&mut self) -> impl Iterator<Item=&mut T>
    {
        self.weights.iter_mut().chain(self.biases.iter_mut())
    }
}

pub struct Predictor<'a>
{
    network: &'a mut NeuralNetwork
}

impl<'a> Drop for Predictor<'a>
{
    fn drop(&mut self)
    {
        self.network.weights_biases_mut().for_each(|layer| layer.enable_gradients());
    }
}

impl<'a> Predictor<'a>
{
    fn new(network: &'a mut NeuralNetwork) -> Self
    {
        network.weights_biases_mut().for_each(|layer| layer.disable_gradients());

        Self{network}
    }

    pub fn feedforward(&mut self, input: Vec<f32>) -> Vec<f32>
    {
        self.network.feedforward_inner(input).as_vec()
    }
}

pub struct TrainingPair
{
    pub input: Vec<f32>,
    pub output: Vec<f32>
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralNetwork
{
    layout: Vec<usize>,
    parameters: ThisWeightsContainer<LayerType>,
    optimizer: CurrentOptimizer
}

impl NeuralNetwork
{
    pub fn new(input_size: usize, learning_rate: f32) -> Self
    {
        let layout = vec![input_size, 256, 256, 256, 3];

        let parameters = ThisWeightsContainer::new_randomized(&layout);

        let mut optimizer = CurrentOptimizer::new(&layout);
        optimizer.set_learning_rate(learning_rate);

        Self{layout, parameters, optimizer}
    }

    pub fn train(&mut self, pairs: impl Iterator<Item=TrainingPair>) -> f32
    {
        self.clear_gradients();

        let mut total_error = 0.0;

        let mut pairs_len = 0;
        for TrainingPair{input, output} in pairs
        {
            let last_size = *self.layout.last().unwrap();
            let output = LayerType::new_diff(
                LayerInnerType::from_raw(output, last_size, 1)
            );

            let predicted_output = self.feedforward_inner(input);

            let error = {
                let mut diff = predicted_output - output;
                diff.pow(2);

                diff.sum()
            };

            total_error += *error.value();

            pairs_len += 1;

            error.calculate_gradients();
        }

        let total_error = total_error / pairs_len as f32;

        let (optimizer_info, optimizer_hyper) = self.optimizer.info_mut();
        self.parameters.weights_biases_mut().zip(optimizer_info.weights_biases_mut())
            .for_each(|(layer, optimizer_info)|
            {
                let gradient = layer.take_gradient().cap_magnitude(1.0);

                let change = CurrentOptimizer::gradient_to_change(
                    optimizer_info,
                    gradient,
                    optimizer_hyper
                );

                *layer.value_mut() -= change;
            });

        total_error
    }

    fn weights_biases_mut(&mut self) -> impl Iterator<Item=&mut LayerType>
    {
        self.parameters.weights_biases_mut()
    }

    pub fn predictor(&mut self) -> Predictor
    {
        Predictor::new(self)
    }

    fn feedforward_inner(&self, input: Vec<f32>) -> LayerType
    {
        let mut layer_input = LayerType::new_diff(
            LayerInnerType::from_raw(input, self.layout[0], 1)
        );

        for layer in 0..(self.layout.len() - 1)
        {
            layer_input = self.feedforward_single(layer_input, layer);
        }

        layer_input
    }

    fn feedforward_single(&self, input: LayerType, layer: usize) -> LayerType
    {
        let mut output = self.parameters.weights[layer].matmulv_add(
            input,
            &self.parameters.biases[layer]
        );

        output.leaky_relu();

        output
    }

    pub fn load(file: impl Read) -> Self
    {
        bincode::deserialize_from(file).unwrap()
    }

    pub fn save(&mut self, path: impl AsRef<Path>)
    {
        let file = File::create(path).unwrap();

        self.clear_gradients();

        bincode::serialize_into(file, self).unwrap();
    }

    fn clear_gradients(&mut self)
    {
        self.weights_biases_mut().for_each(|weight| weight.clear());
    }
}
