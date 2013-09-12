/********************************************************************************
** Form generated from reading UI file 'filteringapp.ui'
**
** Created: Thu Sep 12 22:49:04 2013
**      by: Qt User Interface Compiler version 4.8.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_FILTERINGAPP_H
#define UI_FILTERINGAPP_H

#include <GLDualCamView.h>
#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QComboBox>
#include <QtGui/QFormLayout>
#include <QtGui/QFrame>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QListView>
#include <QtGui/QMainWindow>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QSpinBox>
#include <QtGui/QStatusBar>
#include <QtGui/QToolBar>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_FilteringAppClass
{
public:
    QAction *actionCapture;
    QAction *actionProcess;
    QWidget *centralWidget;
    QGroupBox *groupBox;
    QListView *lvFilters;
    QWidget *layoutWidget;
    QVBoxLayout *verticalLayout_2;
    QSpacerItem *verticalSpacer;
    QVBoxLayout *verticalLayout;
    QPushButton *pbMoveFilterUp;
    QPushButton *pbRemove;
    QPushButton *pbMoveFilterDown;
    QSpacerItem *verticalSpacer_2;
    QWidget *layoutWidget1;
    QHBoxLayout *horizontalLayout;
    QLabel *lblFilterType;
    QComboBox *cmbFilterType;
    QPushButton *pbAddFilter;
    QFrame *frame;
    GLDualCamView *camDual;
    QGroupBox *grpParameters;
    QWidget *formLayoutWidget;
    QFormLayout *formLayout;
    QLabel *lblParameterCaption;
    QSpinBox *spinParameter;
    QSpacerItem *verticalSpacer_3;
    QSpacerItem *verticalSpacer_4;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *FilteringAppClass)
    {
        if (FilteringAppClass->objectName().isEmpty())
            FilteringAppClass->setObjectName(QString::fromUtf8("FilteringAppClass"));
        FilteringAppClass->resize(1079, 664);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(FilteringAppClass->sizePolicy().hasHeightForWidth());
        FilteringAppClass->setSizePolicy(sizePolicy);
        FilteringAppClass->setWindowOpacity(1);
        actionCapture = new QAction(FilteringAppClass);
        actionCapture->setObjectName(QString::fromUtf8("actionCapture"));
        actionCapture->setCheckable(true);
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/FilteringApp/Resources/picture.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionCapture->setIcon(icon);
        actionProcess = new QAction(FilteringAppClass);
        actionProcess->setObjectName(QString::fromUtf8("actionProcess"));
        actionProcess->setCheckable(true);
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/FilteringApp/Resources/3x3_grid_2.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionProcess->setIcon(icon1);
        centralWidget = new QWidget(FilteringAppClass);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        groupBox = new QGroupBox(centralWidget);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        groupBox->setGeometry(QRect(10, 60, 391, 241));
        lvFilters = new QListView(groupBox);
        lvFilters->setObjectName(QString::fromUtf8("lvFilters"));
        lvFilters->setGeometry(QRect(10, 20, 256, 211));
        layoutWidget = new QWidget(groupBox);
        layoutWidget->setObjectName(QString::fromUtf8("layoutWidget"));
        layoutWidget->setGeometry(QRect(280, 20, 97, 211));
        verticalLayout_2 = new QVBoxLayout(layoutWidget);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(6);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        pbMoveFilterUp = new QPushButton(layoutWidget);
        pbMoveFilterUp->setObjectName(QString::fromUtf8("pbMoveFilterUp"));
        pbMoveFilterUp->setDefault(false);
        pbMoveFilterUp->setFlat(false);

        verticalLayout->addWidget(pbMoveFilterUp);

        pbRemove = new QPushButton(layoutWidget);
        pbRemove->setObjectName(QString::fromUtf8("pbRemove"));

        verticalLayout->addWidget(pbRemove);

        pbMoveFilterDown = new QPushButton(layoutWidget);
        pbMoveFilterDown->setObjectName(QString::fromUtf8("pbMoveFilterDown"));
        pbMoveFilterDown->setDefault(false);
        pbMoveFilterDown->setFlat(false);

        verticalLayout->addWidget(pbMoveFilterDown);


        verticalLayout_2->addLayout(verticalLayout);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer_2);

        layoutWidget1 = new QWidget(centralWidget);
        layoutWidget1->setObjectName(QString::fromUtf8("layoutWidget1"));
        layoutWidget1->setGeometry(QRect(10, 10, 474, 30));
        horizontalLayout = new QHBoxLayout(layoutWidget1);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        lblFilterType = new QLabel(layoutWidget1);
        lblFilterType->setObjectName(QString::fromUtf8("lblFilterType"));

        horizontalLayout->addWidget(lblFilterType);

        cmbFilterType = new QComboBox(layoutWidget1);
        cmbFilterType->setObjectName(QString::fromUtf8("cmbFilterType"));
        QSizePolicy sizePolicy1(QSizePolicy::Minimum, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(cmbFilterType->sizePolicy().hasHeightForWidth());
        cmbFilterType->setSizePolicy(sizePolicy1);
        cmbFilterType->setMinimumSize(QSize(300, 0));

        horizontalLayout->addWidget(cmbFilterType);

        pbAddFilter = new QPushButton(layoutWidget1);
        pbAddFilter->setObjectName(QString::fromUtf8("pbAddFilter"));

        horizontalLayout->addWidget(pbAddFilter);

        frame = new QFrame(centralWidget);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setGeometry(QRect(410, 70, 661, 501));
        frame->setStyleSheet(QString::fromUtf8("background-color: blue;"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        camDual = new GLDualCamView(frame);
        camDual->setObjectName(QString::fromUtf8("camDual"));
        camDual->setGeometry(QRect(10, 10, 640, 480));
        grpParameters = new QGroupBox(centralWidget);
        grpParameters->setObjectName(QString::fromUtf8("grpParameters"));
        grpParameters->setGeometry(QRect(10, 310, 391, 80));
        grpParameters->setStyleSheet(QString::fromUtf8("display: none;"));
        formLayoutWidget = new QWidget(grpParameters);
        formLayoutWidget->setObjectName(QString::fromUtf8("formLayoutWidget"));
        formLayoutWidget->setGeometry(QRect(10, 20, 371, 51));
        formLayout = new QFormLayout(formLayoutWidget);
        formLayout->setSpacing(6);
        formLayout->setContentsMargins(11, 11, 11, 11);
        formLayout->setObjectName(QString::fromUtf8("formLayout"));
        formLayout->setContentsMargins(0, 0, 0, 0);
        lblParameterCaption = new QLabel(formLayoutWidget);
        lblParameterCaption->setObjectName(QString::fromUtf8("lblParameterCaption"));

        formLayout->setWidget(1, QFormLayout::LabelRole, lblParameterCaption);

        spinParameter = new QSpinBox(formLayoutWidget);
        spinParameter->setObjectName(QString::fromUtf8("spinParameter"));
        spinParameter->setMaximum(255);
        spinParameter->setValue(100);

        formLayout->setWidget(1, QFormLayout::FieldRole, spinParameter);

        verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        formLayout->setItem(2, QFormLayout::FieldRole, verticalSpacer_3);

        verticalSpacer_4 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        formLayout->setItem(0, QFormLayout::FieldRole, verticalSpacer_4);

        FilteringAppClass->setCentralWidget(centralWidget);
        mainToolBar = new QToolBar(FilteringAppClass);
        mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
        mainToolBar->setMovable(false);
        mainToolBar->setIconSize(QSize(30, 30));
        mainToolBar->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        mainToolBar->setFloatable(false);
        FilteringAppClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(FilteringAppClass);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        statusBar->setStyleSheet(QString::fromUtf8("color: blue; font: 12px;"));
        FilteringAppClass->setStatusBar(statusBar);
#ifndef QT_NO_SHORTCUT
        lblFilterType->setBuddy(cmbFilterType);
#endif // QT_NO_SHORTCUT

        mainToolBar->addSeparator();
        mainToolBar->addAction(actionCapture);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actionProcess);

        retranslateUi(FilteringAppClass);

        QMetaObject::connectSlotsByName(FilteringAppClass);
    } // setupUi

    void retranslateUi(QMainWindow *FilteringAppClass)
    {
        FilteringAppClass->setWindowTitle(QApplication::translate("FilteringAppClass", "OpenCV + CUDA + Qt + C++AMP", 0, QApplication::UnicodeUTF8));
        actionCapture->setText(QApplication::translate("FilteringAppClass", "Capture", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        actionCapture->setToolTip(QApplication::translate("FilteringAppClass", "Starts capturing.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        actionProcess->setText(QApplication::translate("FilteringAppClass", "Process", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        actionProcess->setToolTip(QApplication::translate("FilteringAppClass", "Starts processing the captured images.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        groupBox->setTitle(QApplication::translate("FilteringAppClass", "Filter Chain", 0, QApplication::UnicodeUTF8));
        pbMoveFilterUp->setText(QApplication::translate("FilteringAppClass", "Move Up", 0, QApplication::UnicodeUTF8));
        pbRemove->setText(QApplication::translate("FilteringAppClass", "Remove", 0, QApplication::UnicodeUTF8));
        pbMoveFilterDown->setText(QApplication::translate("FilteringAppClass", "Move Down", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        lblFilterType->setToolTip(QString());
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_STATUSTIP
        lblFilterType->setStatusTip(QApplication::translate("FilteringAppClass", "Select a filter type and press add filter to add filter to the chain.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_STATUSTIP
        lblFilterType->setText(QApplication::translate("FilteringAppClass", "Filte&r Type:", 0, QApplication::UnicodeUTF8));
        pbAddFilter->setText(QApplication::translate("FilteringAppClass", "Add Filter", 0, QApplication::UnicodeUTF8));
        grpParameters->setTitle(QApplication::translate("FilteringAppClass", "Selected Filter Parameters", 0, QApplication::UnicodeUTF8));
        lblParameterCaption->setText(QApplication::translate("FilteringAppClass", "Caption:", 0, QApplication::UnicodeUTF8));
        spinParameter->setPrefix(QString());
    } // retranslateUi

};

namespace Ui {
    class FilteringAppClass: public Ui_FilteringAppClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_FILTERINGAPP_H
