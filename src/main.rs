use std::{
    env,
    io::{self, Write},
    fs,
    f32,
    thread,
    time::Duration,
    fs::File,
    path::PathBuf
};

use sdl2::{
    EventPump,
    event::Event,
    rect::Rect,
    video::{Window, WindowSurfaceRef},
    pixels::Color as SdlColor
};

use image::imageops::FilterType;

use network::{NeuralNetwork, TrainingPair};

mod network;


struct Config
{
    mode: String,
    path: PathBuf,
    input: Option<Vec<PathBuf>>,
    output: PathBuf,
    iterations: usize,
    width: usize,
    height: usize,
    fps: u32,
    seconds: u32,
    learning_rate: f32
}

impl Config
{
    pub fn parse(mut args: impl Iterator<Item=String>) -> Result<Self, String>
    {
        let mut mode = None;
        let mut path = PathBuf::from("network.nn");
        let mut input = None;
        let mut output = PathBuf::from("output.y4m");
        let mut iterations = 1;
        let mut width = 256;
        let mut height = 256;
        let mut fps = 30;
        let mut seconds = 5;
        let mut learning_rate = 0.001;

        while let Some(arg) = args.next()
        {
            match arg.as_ref()
            {
                "-m" | "--mode" =>
                {
                    mode = Some(Self::next_arg(&arg, &mut args)?);
                },
                "-p" | "--path" =>
                {
                    path = PathBuf::from(Self::next_arg(&arg, &mut args)?);
                },
                "-i" | "--input" =>
                {
                    let value = Self::next_arg(&arg, &mut args)?;
                    let paths: Vec<_> = value.split(',').into_iter().map(|path|
                    {
                        PathBuf::from(path.trim())
                    }).collect();

                    input = Some(paths);
                },
                "-o" | "--output" =>
                {
                    output = PathBuf::from(Self::next_arg(&arg, &mut args)?);
                },
                "-f" | "--folder" =>
                {
                    let folder = Self::next_arg(&arg, &mut args)?;

                    let inputs = fs::read_dir(folder).unwrap().filter_map(|maybe_entry|
                    {
                        maybe_entry.ok()
                    }).filter(|entry|
                    {
                        entry.file_type().map(|file_type|
                        {
                            file_type.is_file()
                        }).unwrap_or(false)
                    }).map(|entry| entry.path()).collect();

                    input = Some(inputs);
                },
                "-n" | "--iterations" =>
                {
                    iterations = Self::next_arg(&arg, &mut args)?.parse::<usize>()
                        .map_err(|err| err.to_string())?;
                },
                "--fps" =>
                {
                    fps = Self::next_arg(&arg, &mut args)?.parse::<u32>()
                        .map_err(|err| err.to_string())?;
                },
                "--seconds" =>
                {
                    seconds = Self::next_arg(&arg, &mut args)?.parse::<u32>()
                        .map_err(|err| err.to_string())?;
                },
                "-w" | "--width" =>
                {
                    width = Self::next_arg(&arg, &mut args)?.parse::<usize>()
                        .map_err(|err| err.to_string())?;
                },
                "-h" | "--height" =>
                {
                    height = Self::next_arg(&arg, &mut args)?.parse::<usize>()
                        .map_err(|err| err.to_string())?;
                },
                "-l" | "--learning-rate" =>
                {
                    learning_rate = Self::next_arg(&arg, &mut args)?.parse::<f32>()
                        .map_err(|err| err.to_string())?;
                },
                x => return Err(format!("unexpected argument: {x}"))
            }
        }

        Ok(Self{
            mode: mode.ok_or_else(|| "provide a mode")?,
            path,
            input,
            output,
            iterations,
            width,
            height,
            fps,
            seconds,
            learning_rate
        })
    }

    fn next_arg(arg: &str, mut args: impl Iterator<Item=String>) -> Result<String, String>
    {
        args.next().ok_or_else(|| format!("expected an argument after {arg}"))
    }
}

fn create_input(x: f32, y: f32, image_a: f32) -> Vec<f32>
{
    [x, y].into_iter().flat_map(|v|
    {
        let v = (v * 2.0 - 1.0) * f32::consts::PI;

        [v, v.sin(), v.cos(), (2.0 * v).sin(), (2.0 * v).cos()]
    }).chain([image_a].into_iter()).collect()
}

fn train(config: &Config, network: &mut NeuralNetwork)
{
    let images: Vec<_> = config.input.as_ref().expect("train must have images provided")
        .iter().map(|path|
        {
            let image = image::open(path).unwrap_or_else(|err|
            {
                panic!("error reading {}: {err}", path.display());
            }).resize(config.width as u32, config.height as u32, FilterType::CatmullRom);

            image.into_rgb32f()
        }).collect();

    let images_len = images.len();
    let images: Vec<_> = images.into_iter().zip((0..images_len).map(|index|
    {
        if images_len > 1
        {
            index as f32 / (images_len - 1) as f32
        } else
        {
            0.0
        }
    })).collect();

    for i in 0..config.iterations
    {
        eprintln!("iteration: {i}");

        let total_error: f32 = (0..images.len()).map(|_|
        {
            let (image, image_a) = &images[fastrand::usize(0..images.len())];

            let data = image.enumerate_pixels().map(|(x, y, pixel)|
            {
                let x_f = x as f32 / config.width as f32;
                let y_f = y as f32 / config.height as f32;

                let output: Vec<f32> = pixel.0.into();

                TrainingPair
                {
                    input: create_input(x_f, y_f, *image_a),
                    output
                }
            });

            let error = network.train(data);

            error
        }).sum();

        eprintln!("error: {total_error}");
    }
}

enum Action
{
    Quit,
    MouseMove(f32)
}

#[derive(Debug, Clone, Copy)]
struct RgbColor
{
    pub r: u8,
    pub g: u8,
    pub b: u8
}

#[derive(Debug, Clone, Copy)]
struct YCbCrColor
{
    pub y: u8,
    pub b: u8,
    pub r: u8
}

impl From<RgbColor> for YCbCrColor
{
    fn from(value: RgbColor) -> Self
    {
        let m = [
            [0.299, 0.587, 0.114],
            [-0.169, -0.331, 0.5],
            [0.5, -0.419, -0.081]
        ];

        let mut out = [0.0, 0.0, 0.0];
        let rgb = [value.r as f64, value.g as f64, value.b as f64];
        
        for i in 0..3
        {
            for j in 0..3
            {
                out[i] += rgb[j] * m[i][j];
            }
        }

        let to_u8 = |v: f64|
        {
            v.clamp(0.0, u8::MAX as f64) as u8
        };

        let y = to_u8(out[0]);
        let b = to_u8(out[1] + 128.0);
        let r = to_u8(out[2] + 128.0);

        YCbCrColor{
            y,
            b,
            r
        }
    }
}

trait PixelDrawable
{
    fn set_pixel(&mut self, x: usize, y: usize, color: RgbColor);

    fn draw(&mut self, network: &mut NeuralNetwork, width: usize, height: usize, image_a: f32)
    {
        let mut predictor = network.predictor();
        for y in 0..height
        {
            for x in 0..width
            {
                let x_f = x as f32 / width as f32;
                let y_f = y as f32 / height as f32;

                let pixel = predictor.feedforward(create_input(x_f, y_f, image_a));

                let color_single = |c|
                {
                    (c * u8::MAX as f32) as u8
                };

                let c = RgbColor{
                    r: color_single(pixel[0]),
                    g: color_single(pixel[1]),
                    b: color_single(pixel[2])
                };

                self.set_pixel(x, y, c);
            }
        }
    }
}

struct DisplayWindow<'a>
{
    network: &'a mut NeuralNetwork,
    width: usize,
    height: usize,
    window: Window,
    events: EventPump
}

impl<'a> DisplayWindow<'a>
{
    pub fn new(config: &Config, network: &'a mut NeuralNetwork) -> Self
    {
        let ctx = sdl2::init().unwrap();

        let video = ctx.video().unwrap();
        let events = ctx.event_pump().unwrap();

        let window = video.window("thingy majingy", config.width as u32, config.height as u32)
            .build()
            .unwrap();

        Self{
            width: config.width,
            height: config.height,
            network,
            window,
            events
        }
    }

    pub fn display(&mut self, image_a: f32)
    {
        let surface = self.window.surface(&self.events).unwrap();

        WindowDrawer{surface}.draw(&mut self.network, self.width, self.height, image_a);
    }

    fn handle_events(&mut self) -> Option<Action>
    {
        let mut action = None;
        for event in self.events.poll_iter()
        {
            match event
            {
                Event::Quit{..} => return Some(Action::Quit),
                Event::MouseMotion{x, ..} =>
                {
                    let x_f = if self.width > 1
                    {
                        x as f32 / (self.width - 1) as f32
                    } else
                    {
                        0.0
                    };

                    action = Some(Action::MouseMove(x_f));
                },
                _ => ()
            }
        }

        action
    }

    #[allow(dead_code)]
    fn refresh(&mut self)
    {
        let surface = self.window.surface(&self.events).unwrap();

        surface.update_window().unwrap();
    }
}

struct WindowDrawer<'a>
{
    surface: WindowSurfaceRef<'a>
}

impl<'a> Drop for WindowDrawer<'a>
{
    fn drop(&mut self)
    {
        self.surface.update_window().unwrap();
    }
}

impl<'a> PixelDrawable for WindowDrawer<'a>
{
    fn set_pixel(&mut self, x: usize, y: usize, color: RgbColor)
    {
        let c = SdlColor::RGB(color.r, color.g, color.b);

        self.surface.fill_rect(Rect::new(x as i32, y as i32, 1, 1), c).unwrap();
    }
}

struct SimpleImage
{
    data: Vec<RgbColor>,
    width: usize
}

impl SimpleImage
{
    pub fn new(width: usize, height: usize) -> Self
    {
        Self{data: vec![RgbColor{r: 0, g: 0, b: 0}; width * height], width}
    }
}

impl PixelDrawable for SimpleImage
{
    fn set_pixel(&mut self, x: usize, y: usize, color: RgbColor)
    {
        self.data[y * self.width + x] = color;
    }
}

fn run(config: &Config, network: &mut NeuralNetwork)
{
    let mut window = DisplayWindow::new(config, network);

    let mut image_a = 0.0;

    window.display(image_a);
    loop
    {
        if let Some(action) = window.handle_events()
        {
            match action
            {
                Action::Quit => return,
                Action::MouseMove(x) =>
                {
                    image_a = x;

                    window.display(image_a);
                }
            }
        }

        window.refresh();

        thread::sleep(Duration::from_millis(1000 / 30));
    }
}

fn video(config: &Config, network: &mut NeuralNetwork)
{
    let mut file = File::create(&config.output).unwrap();

    let header = format!("YUV4MPEG2 W{} H{} F{}:1 C444\n", config.width, config.height, config.fps);
    file.write(&header.into_bytes()).unwrap();

    let total_frames = config.seconds * config.fps;
    for i in 0..total_frames
    {
        let image_a = i as f32 / (total_frames - 1) as f32;

        let mut image = SimpleImage::new(config.width, config.height);
        image.draw(network, config.width, config.height, image_a);

        let total_frame_size = config.width * config.height;
        let mut y_plane = Vec::with_capacity(total_frame_size);
        let mut b_plane = Vec::with_capacity(total_frame_size);
        let mut r_plane = Vec::with_capacity(total_frame_size);

        for (_i, color) in image.data.into_iter().enumerate()
        {
            let YCbCrColor{y, b, r} = color.into();

            y_plane.push(y);
            b_plane.push(b);
            r_plane.push(r);
        }

        file.write(b"FRAME\n").unwrap();

        file.write(&y_plane).unwrap();
        file.write(&b_plane).unwrap();
        file.write(&r_plane).unwrap();
    }

    file.flush().unwrap();
}

fn main()
{
    let config = Config::parse(env::args().skip(1)).unwrap();

    match config.mode.as_ref()
    {
        "train" =>
        {
            let mut network = match File::open(&config.path)
            {
                Ok(file) =>
                {
                    NeuralNetwork::load(file)
                },
                Err(err) if err.kind() == io::ErrorKind::NotFound =>
                {
                    let input_size = create_input(0.0, 0.0, 0.0).len();

                    NeuralNetwork::new(input_size, config.learning_rate)
                },
                Err(x) => panic!("error opening path: {}", x)
            };

            train(&config, &mut network);

            network.save(config.path);
        },
        "run" =>
        {
            let mut network = NeuralNetwork::load(File::open(&config.path).unwrap());

            run(&config, &mut network);
        },
        "video" =>
        {
            let mut network = NeuralNetwork::load(File::open(&config.path).unwrap());

            video(&config, &mut network);
        },
        x => panic!("{} isnt a valid mode", x)
    }
}
