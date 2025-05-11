// W aplikacji WPF:
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using System.Collections.Generic;
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
            PlotModel = new PlotModel { Title = "Błąd Średniokwadratowy podczas Treningu" };
            PlotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Błąd Średniokwadratowy" });
            PlotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Epoka" });
            DataContext = this;

            string relativePath = Path.Combine("..", "..", "..", "..", "MLP Neural Network", "bin", "Debug", "net9.0", "wszystkieBledy.txt");

            OdczytajIWyświetlBłędy(relativePath);
        }

        private void OdczytajIWyświetlBłędy(string filePath)
        {
            if (File.Exists(filePath))
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
                PlotModel.InvalidatePlot(true); // Odśwież wykres
            }
            else
            {
                MessageBox.Show($"Plik z historią błędów '{filePath}' nie został znaleziony.");
            }
        }
    }
}