using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Windows;

namespace Neural_MLP_Network
{
    public partial class MainWindow : Window
    {
        public PlotModel PlotModel { get; set; }

        public MainWindow()
        {
            InitializeComponent();

            string info = null;
            string[] args = Environment.GetCommandLineArgs();

            if (args.Length > 1)
            {
                string filePath = args[1];
                if (args.Length > 2)
                {
                    info = args[2];
                }
                OdczytajIWyświetlBłędy(filePath, info);
            }
            else
            {
                MessageBox.Show("Nie podano ścieżki pliku jako argument.");
                PlotModel = new PlotModel { Title = "Błąd podczas treningu" };
            }

            if (PlotModel == null)
            {
                PlotModel = new PlotModel { Title = $"Błąd podczas treningu {info}" };
            }

            PlotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Błąd Średniokwadratowy" });
            PlotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Epoka" });
            DataContext = this;
        }

        private void OdczytajIWyświetlBłędy(string filePath, string info)
        {
            PlotModel = new PlotModel { Title = $"Błąd podczas treningu {info}" };

            if (File.Exists(filePath))
            {
                try
                {
                    List<double> historiaBledow = File.ReadAllLines(filePath)
                        .Select(s => double.Parse(s, CultureInfo.InvariantCulture))
                        .ToList();

                    var lineSeries = new LineSeries
                    {
                        ItemsSource = historiaBledow.Select((blad, indeks) => new DataPoint(indeks * 50, blad)),
                        DataFieldX = "X",
                        DataFieldY = "Y",
                        Title = "Błąd Treningowy"
                    };

                    PlotModel.Series.Add(lineSeries);
                    PlotModel.InvalidatePlot(true);
                }
                catch (FileNotFoundException)
                {
                    MessageBox.Show($"Plik '{filePath}' nie został znaleziony.");
                }
                catch (FormatException)
                {
                    MessageBox.Show($"Nieprawidłowy format danych w pliku '{filePath}'.");
                }
                catch (IOException ex)
                {
                    MessageBox.Show($"Błąd odczytu pliku '{filePath}': {ex.Message}");
                }
            }
            else
            {
                MessageBox.Show($"Plik '{filePath}' nie istnieje.");
            }
        }
    }
}